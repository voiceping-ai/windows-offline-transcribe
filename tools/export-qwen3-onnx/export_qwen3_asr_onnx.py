#!/usr/bin/env python3
"""
Export Qwen3-ASR-0.6B to ONNX format (encoder.onnx + decoder.onnx).

Based on libs/qwen-asr/python_simple_implementation.py (the authoritative reference).

Exports:
  encoder.onnx - Audio Encoder + Multi-Modal Projector
    Input:  mel (float32, [1, 128, T])
    Output: audio_features (float32, [1, N, 1024])

  decoder.onnx - LLM Decoder with KV cache
    Inputs:  inputs_embeds (float32, [1, S, 1024]),
             position_ids (int64, [1, S]),
             past_key_values_0_key..past_key_values_27_value (float32, [1, 8, L, 128])
    Outputs: logits (float32, [1, S, 151936]),
             present_key_values_0_key..present_key_values_27_value

Also exports:
  embed_tokens.bin - Raw float32 embedding matrix (~590MB)
  vocab.json, merges.txt - Copied from source model

Usage:
    pip install -r requirements.txt
    python export_qwen3_asr_onnx.py --model-dir /path/to/Qwen3-ASR-0.6B --output-dir ./output
    python export_qwen3_asr_onnx.py --model-id Qwen/Qwen3-ASR-0.6B --output-dir ./output
"""

import argparse
import json
import math
import os
import shutil
import sys
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ============================================================================
# Config + Weight loading (from python_simple_implementation.py)
# ============================================================================

def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["thinker_config"]
    ac = tc["audio_config"]
    txc = tc["text_config"]
    return {
        "enc_d_model": ac["d_model"],
        "enc_layers": ac["encoder_layers"],
        "enc_heads": ac["encoder_attention_heads"],
        "enc_ffn_dim": ac["encoder_ffn_dim"],
        "enc_output_dim": ac["output_dim"],
        "enc_downsample_hidden": ac["downsample_hidden_size"],
        "enc_num_mel_bins": ac["num_mel_bins"],
        "enc_max_source_pos": ac["max_source_positions"],
        "enc_n_window": ac["n_window"],
        "enc_n_window_infer": ac["n_window_infer"],
        "enc_conv_chunksize": ac.get("conv_chunksize", 500),
        "dec_hidden_size": txc["hidden_size"],
        "dec_layers": txc["num_hidden_layers"],
        "dec_heads": txc["num_attention_heads"],
        "dec_kv_heads": txc["num_key_value_heads"],
        "dec_head_dim": txc["head_dim"],
        "dec_intermediate": txc["intermediate_size"],
        "dec_rms_norm_eps": txc["rms_norm_eps"],
        "dec_rope_theta": txc["rope_theta"],
        "dec_vocab_size": txc["vocab_size"],
    }


class MultiSafetensors:
    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            self.files = {}
            for shard in shard_files:
                path = os.path.join(model_dir, shard)
                self.files[shard] = safe_open(path, framework="pt")
            self.weight_map = index["weight_map"]
        else:
            self.files = {"model.safetensors": safe_open(single_path, framework="pt")}
            self.weight_map = None

    def get_tensor(self, name):
        if self.weight_map:
            shard = self.weight_map[name]
            return self.files[shard].get_tensor(name)
        for sf in self.files.values():
            try:
                return sf.get_tensor(name)
            except Exception:
                continue
        raise KeyError(f"Weight not found: {name}")


def get_weight(sf, name):
    t = sf.get_tensor(name)
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t


# ============================================================================
# Encoder Module for ONNX export
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, channels, max_timescale=10000):
        super().__init__()
        self.channels = channels
        log_ts = math.log(max_timescale) / (channels // 2 - 1)
        inv_ts = torch.exp(-log_ts * torch.arange(channels // 2, dtype=torch.float32))
        self.register_buffer("inv_timescales", inv_ts)

    def forward(self, length):
        scaled = torch.arange(length, dtype=torch.float32, device=self.inv_timescales.device).unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


class EncoderAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        # Self-attention
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        B, S, D = x_norm.shape
        q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.out_proj(attn_out)
        # FFN
        residual = x
        x_norm = self.final_layer_norm(x)
        x = residual + self.fc2(F.gelu(self.fc1(x_norm)))
        return x


class Qwen3ASREncoder(nn.Module):
    """Audio encoder + multi-modal projector, matching python_simple_implementation.py.

    For ONNX export, we process ALL mel frames as a single batch (no chunking loop).
    The C# caller is responsible for chunking if needed, or we can pad to multiples of 100.
    For simplicity, we process the full mel in one shot - the Conv2D handles it.
    """
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["enc_d_model"]
        n_layers = cfg["enc_layers"]
        n_heads = cfg["enc_heads"]
        ffn_dim = cfg["enc_ffn_dim"]
        self.d_model = d_model
        self.n_heads = n_heads

        # Conv2D stem
        self.conv2d1 = nn.Conv2d(1, cfg["enc_downsample_hidden"], 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(cfg["enc_downsample_hidden"], cfg["enc_downsample_hidden"], 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(cfg["enc_downsample_hidden"], cfg["enc_downsample_hidden"], 3, stride=2, padding=1)

        # After 3x stride-2 convs on 128 mel bins: 128 -> 64 -> 32 -> 16
        # So conv output channels = downsample_hidden * 16
        conv_out_dim = cfg["enc_downsample_hidden"] * 16  # 480 * 16 = 7680
        self.conv_out = nn.Linear(conv_out_dim, d_model, bias=False)

        # Position embedding
        self.pos_emb = SinusoidalPositionEmbedding(d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            EncoderAttentionLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)
        ])

        # Final LN
        self.ln_post = nn.LayerNorm(d_model)

        # Projector (encoder output_dim -> decoder hidden_size)
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, cfg["dec_hidden_size"])

    def forward(self, mel):
        """mel: [1, 128, T] -> audio_features: [1, N, dec_hidden_size]"""
        # Conv stem
        x = mel.unsqueeze(1)  # [1, 1, 128, T]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # [1, C, F, T'] -> [1, T', C*F]
        B, C, Fr, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fr)

        # Linear to d_model
        x = self.conv_out(x)  # [1, T', d_model]
        seq_len = x.shape[1]

        # Add sinusoidal pos embeddings
        pos = self.pos_emb(seq_len)  # [T', d_model]
        x = x + pos.unsqueeze(0)

        # Transformer layers (full attention within batch dim=1, no windowing for ONNX)
        for layer in self.layers:
            x = layer(x)

        # Final LN + projector
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        return x  # [1, N, dec_hidden_size]


# ============================================================================
# Decoder Module for ONNX export
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # All weights are float32 (converted from bf16 during loading),
        # so we avoid .to(x.dtype) which produces problematic Cast nodes in ONNX.
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.head_dim = head_dim

    def forward(self, position_ids):
        """position_ids: [1, S] -> cos [1, S, head_dim], sin [1, S, head_dim]"""
        # [S] x [head_dim/2] -> [S, head_dim/2]
        pos = position_ids.squeeze(0).float()
        angles = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([angles, angles], dim=-1)  # [S, head_dim]
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def apply_rotary_pos_emb(x, cos, sin):
    """x: [B, n_heads, S, head_dim], cos/sin: [1, S, head_dim]"""
    cos = cos.unsqueeze(1)  # [1, 1, S, head_dim]
    sin = sin.unsqueeze(1)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim, intermediate_size, eps):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = n_heads // n_kv_heads

        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, h, cos, sin, past_key, past_value):
        """
        h: [1, S, hidden_size]
        cos, sin: [1, S, head_dim]
        past_key, past_value: [1, n_kv_heads, L, head_dim]
        Returns: h, present_key, present_value
        """
        B, S, D = h.shape
        residual = h
        x = self.input_layernorm(h)

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Per-head Q/K norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # [B, S, n_heads, head_dim] -> [B, n_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Concat with past KV cache
        k = torch.cat([past_key, k], dim=2)
        v = torch.cat([past_value, v], dim=2)

        present_key = k
        present_value = v

        # GQA expansion
        if self.gqa_ratio > 1:
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
            v = v.repeat_interleave(self.gqa_ratio, dim=1)

        # Scaled dot-product attention (causal)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        total_len = k.shape[2]
        # Create causal mask: each query position can attend to itself and all previous KV positions
        causal_mask = torch.triu(
            torch.full((S, total_len), float("-inf"), device=h.device, dtype=h.dtype),
            diagonal=total_len - S + 1
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)

        h = residual + self.o_proj(attn_out)

        # FFN
        residual = h
        x = self.post_attention_layernorm(h)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = residual + self.down_proj(gate * up)

        return h, present_key, present_value


class Qwen3ASRDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = cfg["dec_hidden_size"]
        n_layers = cfg["dec_layers"]
        n_heads = cfg["dec_heads"]
        n_kv_heads = cfg["dec_kv_heads"]
        head_dim = cfg["dec_head_dim"]
        intermediate = cfg["dec_intermediate"]
        eps = cfg["dec_rms_norm_eps"]
        vocab_size = cfg["dec_vocab_size"]

        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.rotary_emb = RotaryEmbedding(head_dim, cfg["dec_rope_theta"])
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, n_heads, n_kv_heads, head_dim, intermediate, eps)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, inputs_embeds, position_ids, *past_kv_flat):
        """
        inputs_embeds: [1, S, hidden_size]
        position_ids: [1, S]
        past_kv_flat: 28 * 2 tensors (key, value) each [1, n_kv_heads, L, head_dim]
        Returns: logits [1, S, vocab_size], 28 * 2 present KV tensors
        """
        cos, sin = self.rotary_emb(position_ids)

        h = inputs_embeds
        present_kvs = []
        for i, layer in enumerate(self.layers):
            past_key = past_kv_flat[i * 2]
            past_value = past_kv_flat[i * 2 + 1]
            h, present_key, present_value = layer(h, cos, sin, past_key, past_value)
            present_kvs.append(present_key)
            present_kvs.append(present_value)

        h = self.norm(h)
        logits = self.lm_head(h.float())

        return (logits, *present_kvs)


# ============================================================================
# Weight loading into modules
# ============================================================================

def load_encoder_weights(encoder, sf, cfg):
    """Load weights from safetensors into the encoder module."""
    prefix = "thinker.audio_tower"

    # Conv stem
    encoder.conv2d1.weight.data = get_weight(sf, f"{prefix}.conv2d1.weight")
    encoder.conv2d1.bias.data = get_weight(sf, f"{prefix}.conv2d1.bias")
    encoder.conv2d2.weight.data = get_weight(sf, f"{prefix}.conv2d2.weight")
    encoder.conv2d2.bias.data = get_weight(sf, f"{prefix}.conv2d2.bias")
    encoder.conv2d3.weight.data = get_weight(sf, f"{prefix}.conv2d3.weight")
    encoder.conv2d3.bias.data = get_weight(sf, f"{prefix}.conv2d3.bias")
    encoder.conv_out.weight.data = get_weight(sf, f"{prefix}.conv_out.weight")

    # Transformer layers
    for i in range(cfg["enc_layers"]):
        lp = f"{prefix}.layers.{i}"
        layer = encoder.layers[i]
        layer.self_attn_layer_norm.weight.data = get_weight(sf, f"{lp}.self_attn_layer_norm.weight")
        layer.self_attn_layer_norm.bias.data = get_weight(sf, f"{lp}.self_attn_layer_norm.bias")
        layer.q_proj.weight.data = get_weight(sf, f"{lp}.self_attn.q_proj.weight")
        layer.q_proj.bias.data = get_weight(sf, f"{lp}.self_attn.q_proj.bias")
        layer.k_proj.weight.data = get_weight(sf, f"{lp}.self_attn.k_proj.weight")
        layer.k_proj.bias.data = get_weight(sf, f"{lp}.self_attn.k_proj.bias")
        layer.v_proj.weight.data = get_weight(sf, f"{lp}.self_attn.v_proj.weight")
        layer.v_proj.bias.data = get_weight(sf, f"{lp}.self_attn.v_proj.bias")
        layer.out_proj.weight.data = get_weight(sf, f"{lp}.self_attn.out_proj.weight")
        layer.out_proj.bias.data = get_weight(sf, f"{lp}.self_attn.out_proj.bias")
        layer.final_layer_norm.weight.data = get_weight(sf, f"{lp}.final_layer_norm.weight")
        layer.final_layer_norm.bias.data = get_weight(sf, f"{lp}.final_layer_norm.bias")
        layer.fc1.weight.data = get_weight(sf, f"{lp}.fc1.weight")
        layer.fc1.bias.data = get_weight(sf, f"{lp}.fc1.bias")
        layer.fc2.weight.data = get_weight(sf, f"{lp}.fc2.weight")
        layer.fc2.bias.data = get_weight(sf, f"{lp}.fc2.bias")

    # Final LN
    encoder.ln_post.weight.data = get_weight(sf, f"{prefix}.ln_post.weight")
    encoder.ln_post.bias.data = get_weight(sf, f"{prefix}.ln_post.bias")

    # Projector
    encoder.proj1.weight.data = get_weight(sf, f"{prefix}.proj1.weight")
    encoder.proj1.bias.data = get_weight(sf, f"{prefix}.proj1.bias")
    encoder.proj2.weight.data = get_weight(sf, f"{prefix}.proj2.weight")
    encoder.proj2.bias.data = get_weight(sf, f"{prefix}.proj2.bias")

    print(f"Loaded encoder weights ({cfg['enc_layers']} layers)", file=sys.stderr)


def load_decoder_weights(decoder, sf, cfg):
    """Load weights from safetensors into the decoder module."""
    prefix = "thinker.model"

    for i in range(cfg["dec_layers"]):
        lp = f"{prefix}.layers.{i}"
        layer = decoder.layers[i]
        layer.input_layernorm.weight.data = get_weight(sf, f"{lp}.input_layernorm.weight")
        layer.q_proj.weight.data = get_weight(sf, f"{lp}.self_attn.q_proj.weight")
        layer.k_proj.weight.data = get_weight(sf, f"{lp}.self_attn.k_proj.weight")
        layer.v_proj.weight.data = get_weight(sf, f"{lp}.self_attn.v_proj.weight")
        layer.o_proj.weight.data = get_weight(sf, f"{lp}.self_attn.o_proj.weight")
        layer.q_norm.weight.data = get_weight(sf, f"{lp}.self_attn.q_norm.weight")
        layer.k_norm.weight.data = get_weight(sf, f"{lp}.self_attn.k_norm.weight")
        layer.post_attention_layernorm.weight.data = get_weight(sf, f"{lp}.post_attention_layernorm.weight")
        layer.gate_proj.weight.data = get_weight(sf, f"{lp}.mlp.gate_proj.weight")
        layer.up_proj.weight.data = get_weight(sf, f"{lp}.mlp.up_proj.weight")
        layer.down_proj.weight.data = get_weight(sf, f"{lp}.mlp.down_proj.weight")
        if (i + 1) % 8 == 0:
            print(f"  Decoder layer {i+1}/{cfg['dec_layers']} loaded", file=sys.stderr)

    decoder.norm.weight.data = get_weight(sf, f"{prefix}.norm.weight")
    decoder.lm_head.weight.data = get_weight(sf, "thinker.lm_head.weight")

    print(f"Loaded decoder weights ({cfg['dec_layers']} layers)", file=sys.stderr)


# ============================================================================
# Export functions
# ============================================================================

def export_encoder(encoder, cfg, output_dir):
    """Export encoder to ONNX."""
    encoder.eval()
    output_path = os.path.join(output_dir, "encoder.onnx")

    # Dummy input: 1 batch, 128 mel bins, 200 frames (2 chunks of 100)
    dummy_mel = torch.randn(1, 128, 200)

    print("Exporting encoder to ONNX...", file=sys.stderr)
    torch.onnx.export(
        encoder,
        (dummy_mel,),
        output_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["audio_features"],
        dynamic_axes={
            "mel": {2: "num_frames"},
            "audio_features": {1: "num_tokens"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Encoder exported to {output_path}", file=sys.stderr)

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = encoder(dummy_mel).numpy()
    ort_out = sess.run(None, {"mel": dummy_mel.numpy()})[0]
    diff = np.abs(pt_out - ort_out).max()
    print(f"Encoder validation: max diff = {diff:.6e}", file=sys.stderr)
    assert diff < 1e-4, f"Encoder validation failed: max diff {diff}"

    return output_path


def export_decoder(decoder, cfg, output_dir):
    """Export decoder to ONNX with KV cache."""
    decoder.eval()
    output_path = os.path.join(output_dir, "decoder.onnx")

    n_layers = cfg["dec_layers"]
    n_kv_heads = cfg["dec_kv_heads"]
    head_dim = cfg["dec_head_dim"]
    hidden_size = cfg["dec_hidden_size"]

    # Dummy inputs
    seq_len = 5
    past_len = 0
    dummy_embeds = torch.randn(1, seq_len, hidden_size)
    dummy_pos = torch.arange(past_len, past_len + seq_len, dtype=torch.long).unsqueeze(0)

    # Past KV cache (empty for prefill)
    past_kvs = []
    for i in range(n_layers):
        past_kvs.append(torch.zeros(1, n_kv_heads, past_len, head_dim))  # key
        past_kvs.append(torch.zeros(1, n_kv_heads, past_len, head_dim))  # value

    input_names = ["inputs_embeds", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }

    for i in range(n_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present_key_values.{i}.key")
        output_names.append(f"present_key_values.{i}.value")
        dynamic_axes[f"past_key_values.{i}.key"] = {2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {2: "past_len"}
        dynamic_axes[f"present_key_values.{i}.key"] = {2: "total_len"}
        dynamic_axes[f"present_key_values.{i}.value"] = {2: "total_len"}

    args = (dummy_embeds, dummy_pos, *past_kvs)

    print("Exporting decoder to ONNX...", file=sys.stderr)
    torch.onnx.export(
        decoder,
        args,
        output_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Decoder exported to {output_path}", file=sys.stderr)

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_outputs = decoder(*args)
    pt_logits = pt_outputs[0].numpy()

    ort_inputs = {"inputs_embeds": dummy_embeds.numpy(), "position_ids": dummy_pos.numpy()}
    for i in range(n_layers):
        ort_inputs[f"past_key_values.{i}.key"] = past_kvs[i * 2].numpy()
        ort_inputs[f"past_key_values.{i}.value"] = past_kvs[i * 2 + 1].numpy()
    ort_outputs = sess.run(None, ort_inputs)
    ort_logits = ort_outputs[0]

    diff = np.abs(pt_logits - ort_logits).max()
    print(f"Decoder validation: max diff = {diff:.6e}", file=sys.stderr)
    assert diff < 1e-4, f"Decoder validation failed: max diff {diff}"

    return output_path


def export_embeddings(sf, output_dir):
    """Export embedding matrix as raw float32 binary."""
    output_path = os.path.join(output_dir, "embed_tokens.bin")
    embed = get_weight(sf, "thinker.model.embed_tokens.weight")
    print(f"Embedding matrix: {embed.shape} ({embed.numel() * 4 / 1024 / 1024:.1f} MB)", file=sys.stderr)
    embed_np = embed.numpy().astype(np.float32)
    embed_np.tofile(output_path)
    print(f"Embeddings exported to {output_path}", file=sys.stderr)
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR to ONNX")
    parser.add_argument("--model-dir", type=str, help="Path to local model directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-ASR-0.6B",
                        help="HuggingFace model ID (downloads if --model-dir not given)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for ONNX files")
    parser.add_argument("--skip-validation", action="store_true", help="Skip ONNX validation")
    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.model_id}...", file=sys.stderr)
        model_dir = snapshot_download(args.model_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and weights
    cfg = load_config(model_dir)
    print(f"Config: enc_d={cfg['enc_d_model']}, enc_layers={cfg['enc_layers']}, "
          f"dec_hidden={cfg['dec_hidden_size']}, dec_layers={cfg['dec_layers']}", file=sys.stderr)

    sf = MultiSafetensors(model_dir)

    # Build and export encoder
    print("\n=== Encoder ===", file=sys.stderr)
    encoder = Qwen3ASREncoder(cfg)
    load_encoder_weights(encoder, sf, cfg)
    export_encoder(encoder, cfg, args.output_dir)
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build and export decoder
    print("\n=== Decoder ===", file=sys.stderr)
    decoder = Qwen3ASRDecoder(cfg)
    load_decoder_weights(decoder, sf, cfg)
    export_decoder(decoder, cfg, args.output_dir)
    del decoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Export embeddings
    print("\n=== Embeddings ===", file=sys.stderr)
    export_embeddings(sf, args.output_dir)

    # Copy tokenizer files
    print("\n=== Tokenizer files ===", file=sys.stderr)
    for fname in ["vocab.json", "merges.txt"]:
        src = os.path.join(model_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname}", file=sys.stderr)
        else:
            print(f"WARNING: {fname} not found in model dir", file=sys.stderr)

    print(f"\nExport complete. Files in {args.output_dir}:", file=sys.stderr)
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  {f}: {size / 1024 / 1024:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
