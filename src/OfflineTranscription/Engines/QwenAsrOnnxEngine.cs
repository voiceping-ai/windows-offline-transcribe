using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OfflineTranscription.Audio;
using OfflineTranscription.Interfaces;
using OfflineTranscription.Models;
using OfflineTranscription.Tokenizers;

namespace OfflineTranscription.Engines;

/// <summary>
/// Qwen3-ASR engine via ONNX Runtime with DirectML GPU acceleration (CPU fallback).
/// Runs exported encoder.onnx + decoder.onnx models.
/// Thread-safe: all inference guarded by SemaphoreSlim.
/// </summary>
public sealed class QwenAsrOnnxEngine : IASREngine
{
    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private float[]? _embedTokensMatrix; // [vocab_size, hidden_size]
    private QwenTokenizer? _tokenizer;
    private int _vocabSize;
    private int _hiddenSize;
    private int _numLayers;
    private int _numKvHeads;
    private int _headDim;
    private string _provider = "cpu";

    private readonly SemaphoreSlim _sem = new(1, 1);
    private bool _disposed;

    // Constants from model config (0.6B variant)
    private const int DefaultNumLayers = 28;
    private const int DefaultNumKvHeads = 8;
    private const int DefaultHeadDim = 128;
    private const int DefaultHiddenSize = 1024;
    private const int DefaultVocabSize = 151936;
    private const int MaxNewTokens = 500;

    public bool IsLoaded => _encoderSession != null && _decoderSession != null;
    public bool IsStreaming => false;

    public async Task<bool> LoadModelAsync(string modelDir, CancellationToken ct = default)
    {
        await _sem.WaitAsync(ct);
        try
        {
            Release_Internal();

            return await Task.Run(() =>
            {
                var encoderPath = Path.Combine(modelDir, "encoder.onnx");
                var decoderPath = Path.Combine(modelDir, "decoder.onnx");
                var vocabPath = Path.Combine(modelDir, "vocab.json");
                var embedPath = Path.Combine(modelDir, "embed_tokens.bin");

                if (!File.Exists(encoderPath) || !File.Exists(decoderPath))
                {
                    Debug.WriteLine("[QwenAsrOnnx] Missing encoder.onnx or decoder.onnx");
                    return false;
                }

                // Try DirectML first, fall back to CPU
                _provider = "cpu";
                SessionOptions? sessionOpts = null;
                try
                {
                    try
                    {
                        sessionOpts = CreateSessionOptions("directml");
                        // Probe with encoder (smaller model)
                        using var probe = new InferenceSession(encoderPath, sessionOpts);
                        _provider = "directml";
                        Debug.WriteLine("[QwenAsrOnnx] DirectML provider available");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"[QwenAsrOnnx] DirectML probe failed: {ex.Message}");
                        sessionOpts?.Dispose();
                        sessionOpts = null;
                    }

                    if (sessionOpts == null)
                    {
                        sessionOpts = CreateSessionOptions("cpu");
                        Debug.WriteLine("[QwenAsrOnnx] Using CPU provider");
                    }

                    _encoderSession = new InferenceSession(encoderPath, sessionOpts);
                    _decoderSession = new InferenceSession(decoderPath, sessionOpts);

                    if (!File.Exists(vocabPath))
                    {
                        Debug.WriteLine("[QwenAsrOnnx] WARNING: vocab.json not found");
                        Release_Internal();
                        return false;
                    }

                    _tokenizer = new QwenTokenizer(vocabPath);
                    Debug.WriteLine("[QwenAsrOnnx] Tokenizer loaded");

                    if (!File.Exists(embedPath))
                    {
                        Debug.WriteLine("[QwenAsrOnnx] WARNING: embed_tokens.bin not found");
                        Release_Internal();
                        return false;
                    }

                    var bytes = File.ReadAllBytes(embedPath);
                    if (bytes.Length == 0 || bytes.Length % sizeof(float) != 0)
                    {
                        Debug.WriteLine("[QwenAsrOnnx] Invalid embed_tokens.bin size");
                        Release_Internal();
                        return false;
                    }

                    _embedTokensMatrix = new float[bytes.Length / sizeof(float)];
                    Buffer.BlockCopy(bytes, 0, _embedTokensMatrix, 0, bytes.Length);

                    // Infer dimensions: vocab_size * hidden_size = total floats
                    _vocabSize = DefaultVocabSize;
                    if (_embedTokensMatrix.Length % _vocabSize != 0)
                    {
                        Debug.WriteLine(
                            $"[QwenAsrOnnx] Invalid embedding dimensions: total={_embedTokensMatrix.Length}, vocab={_vocabSize}");
                        Release_Internal();
                        return false;
                    }

                    _hiddenSize = _embedTokensMatrix.Length / _vocabSize;
                    if (_hiddenSize != DefaultHiddenSize)
                    {
                        Debug.WriteLine(
                            $"[QwenAsrOnnx] Unexpected hidden size: {_hiddenSize}, expected {DefaultHiddenSize}");
                        Release_Internal();
                        return false;
                    }

                    Debug.WriteLine($"[QwenAsrOnnx] Embeddings loaded: [{_vocabSize}, {_hiddenSize}]");

                    // Set decoder dimensions
                    _numLayers = DefaultNumLayers;
                    _numKvHeads = DefaultNumKvHeads;
                    _headDim = DefaultHeadDim;

                    Debug.WriteLine($"[QwenAsrOnnx] Model loaded (provider={_provider})");
                    return true;
                }
                catch (OperationCanceledException)
                {
                    Release_Internal();
                    throw;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[QwenAsrOnnx] Model load failed: {ex.Message}");
                    Release_Internal();
                    return false;
                }
                finally
                {
                    try
                    {
                        sessionOpts?.Dispose();
                    }
                    catch
                    {
                        // SessionOptions.Dispose should not fail in practice; ignore if it does.
                    }
                }
            }, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Debug.WriteLine($"[QwenAsrOnnx] Model load failed: {ex.Message}");
            Release_Internal();
            return false;
        }
        finally
        {
            _sem.Release();
        }
    }

    public async Task<ASRResult> TranscribeAsync(
        float[] audioSamples,
        int numThreads,
        string language,
        CancellationToken ct = default)
    {
        if (!IsLoaded) return ASRResult.Empty;

        await _sem.WaitAsync(ct);
        try
        {
            if (_encoderSession == null || _decoderSession == null)
                return ASRResult.Empty;

            var sw = Stopwatch.StartNew();

            var result = await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();

                // 1. Compute mel spectrogram
                var (melData, melBins, numFrames) = MelSpectrogram.Compute(audioSamples);
                if (numFrames == 0) return ASRResult.Empty;

                Debug.WriteLine($"[QwenAsrOnnx] Mel: [{melBins}, {numFrames}] from {audioSamples.Length} samples");

                // 2. Run encoder
                var audioFeatures = RunEncoder(melData, melBins, numFrames);
                int numAudioTokens = audioFeatures.Length / _hiddenSize;
                Debug.WriteLine($"[QwenAsrOnnx] Encoder output: [{numAudioTokens}, {_hiddenSize}]");

                ct.ThrowIfCancellationRequested();

                // 3. Build prompt embeddings
                var inputEmbeds = BuildPromptEmbeddings(audioFeatures, numAudioTokens);
                int promptLen = QwenTokenizer.PromptPrefix.Length + numAudioTokens + QwenTokenizer.PromptSuffix.Length;
                Debug.WriteLine($"[QwenAsrOnnx] Prompt: {promptLen} tokens ({numAudioTokens} audio)");

                // 4. Prefill + autoregressive decode
                var generatedTokens = Decode(inputEmbeds, promptLen, ct);
                Debug.WriteLine($"[QwenAsrOnnx] Generated {generatedTokens.Count} tokens");

                // 5. Decode tokens to text
                var text = _tokenizer!.Decode(generatedTokens);

                // Parse ASR output format
                if (text.Contains("<asr_text>"))
                    text = text.Split("<asr_text>", 2)[1];
                text = text.Trim();

                if (string.IsNullOrWhiteSpace(text))
                    return ASRResult.Empty;

                var segments = new[] { new ASRSegment(text) };
                return new ASRResult(text, segments);
            }, ct);

            sw.Stop();
            return result with { InferenceTimeMs = sw.Elapsed.TotalMilliseconds };
        }
        finally
        {
            _sem.Release();
        }
    }

    /// <summary>Run encoder: mel [128, T] -> audio features [N, hidden_size].</summary>
    private float[] RunEncoder(float[] melData, int melBins, int numFrames)
    {
        // Input shape: [1, 128, numFrames]
        var melTensor = new DenseTensor<float>(melData, new[] { 1, melBins, numFrames });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mel", melTensor)
        };

        using var results = _encoderSession!.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Output: [1, N, hidden_size] -> copy to flat array
        var dims = output.Dimensions;
        int n = dims[1];
        int d = dims[2];
        var features = new float[n * d];
        int idx = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                features[idx++] = output[0, i, j];

        return features;
    }

    /// <summary>
    /// Build full prompt embeddings: prefix tokens + audio features + suffix tokens.
    /// Returns flat float array [promptLen * hiddenSize].
    /// </summary>
    private float[] BuildPromptEmbeddings(float[] audioFeatures, int numAudioTokens)
    {
        var prefix = QwenTokenizer.PromptPrefix;
        var suffix = QwenTokenizer.PromptSuffix;
        int promptLen = prefix.Length + numAudioTokens + suffix.Length;
        var embeds = new float[promptLen * _hiddenSize];

        int pos = 0;

        // Embed prefix tokens
        foreach (var tokenId in prefix)
        {
            EmbedToken(tokenId, embeds, pos * _hiddenSize);
            pos++;
        }

        // Insert audio features (replacing audio_pad token embeddings)
        Array.Copy(audioFeatures, 0, embeds, pos * _hiddenSize, numAudioTokens * _hiddenSize);
        pos += numAudioTokens;

        // Embed suffix tokens
        foreach (var tokenId in suffix)
        {
            EmbedToken(tokenId, embeds, pos * _hiddenSize);
            pos++;
        }

        return embeds;
    }

    /// <summary>Copy embedding for a single token into dest array at given offset.</summary>
    private void EmbedToken(int tokenId, float[] dest, int destOffset)
    {
        int srcOffset = tokenId * _hiddenSize;
        Array.Copy(_embedTokensMatrix!, srcOffset, dest, destOffset, _hiddenSize);
    }

    /// <summary>
    /// Prefill with full prompt, then autoregressive decode until EOS or max tokens.
    /// </summary>
    private List<int> Decode(float[] promptEmbeds, int promptLen, CancellationToken ct)
    {
        // Prefill: run decoder with full prompt
        int pastLen = 0;
        var (logits, kvCache) = RunDecoder(promptEmbeds, promptLen, pastLen);
        pastLen = promptLen;

        // Get first token from last logit position
        int firstToken = ArgMax(logits, (promptLen - 1) * _vocabSize, _vocabSize);
        var generated = new List<int> { firstToken };
        Debug.WriteLine($"[QwenAsrOnnx] First token: {firstToken}");

        if (QwenTokenizer.EosTokenIds.Contains(firstToken))
            return generated;

        // Autoregressive decode
        for (int step = 0; step < MaxNewTokens - 1; step++)
        {
            ct.ThrowIfCancellationRequested();

            int lastToken = generated[^1];

            // Embed the last generated token
            var tokenEmbed = new float[_hiddenSize];
            EmbedToken(lastToken, tokenEmbed, 0);

            // Run decoder with single token + KV cache
            var (stepLogits, newKvCache) = RunDecoderStep(tokenEmbed, pastLen, kvCache);
            kvCache = newKvCache;
            pastLen++;

            int nextToken = ArgMax(stepLogits, 0, _vocabSize);
            generated.Add(nextToken);

            if (QwenTokenizer.EosTokenIds.Contains(nextToken))
                break;
        }

        // Remove EOS tokens from end
        while (generated.Count > 0 && QwenTokenizer.EosTokenIds.Contains(generated[^1]))
            generated.RemoveAt(generated.Count - 1);

        return generated;
    }

    /// <summary>Run decoder with full sequence (prefill).</summary>
    private (float[] logits, float[][] kvCache) RunDecoder(
        float[] embedsFlat, int seqLen, int pastLen)
    {
        var embedsTensor = new DenseTensor<float>(embedsFlat, new[] { 1, seqLen, _hiddenSize });

        var posIds = new long[seqLen];
        for (int i = 0; i < seqLen; i++)
            posIds[i] = pastLen + i;
        var posTensor = new DenseTensor<long>(posIds, new[] { 1, seqLen });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", embedsTensor),
            NamedOnnxValue.CreateFromTensor("position_ids", posTensor)
        };

        // Empty past KV cache for prefill
        for (int i = 0; i < _numLayers; i++)
        {
            var emptyKey = new DenseTensor<float>(new[] { 1, _numKvHeads, 0, _headDim });
            var emptyVal = new DenseTensor<float>(new[] { 1, _numKvHeads, 0, _headDim });
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.key", emptyKey));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.value", emptyVal));
        }

        using var results = _decoderSession!.Run(inputs);
        var resultList = results.ToList();

        // Extract logits
        var logitsTensor = resultList[0].AsTensor<float>();
        var logits = new float[seqLen * _vocabSize];
        CopyTensorToFlat(logitsTensor, logits);

        // Extract present KV cache (28 layers * 2)
        var kvCache = new float[_numLayers * 2][];
        for (int i = 0; i < _numLayers * 2; i++)
        {
            var kv = resultList[1 + i].AsTensor<float>();
            kvCache[i] = TensorToArray(kv);
        }

        return (logits, kvCache);
    }

    /// <summary>Run decoder with single token (autoregressive step).</summary>
    private (float[] logits, float[][] kvCache) RunDecoderStep(
        float[] tokenEmbed, int position, float[][] prevKvCache)
    {
        var embedsTensor = new DenseTensor<float>(tokenEmbed, new[] { 1, 1, _hiddenSize });
        var posTensor = new DenseTensor<long>(new[] { (long)position }, new[] { 1, 1 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", embedsTensor),
            NamedOnnxValue.CreateFromTensor("position_ids", posTensor)
        };

        // Feed previous KV cache
        int kvSeqLen = prevKvCache[0].Length / (_numKvHeads * _headDim);
        for (int i = 0; i < _numLayers; i++)
        {
            var keyData = prevKvCache[i * 2];
            var valData = prevKvCache[i * 2 + 1];
            var keyTensor = new DenseTensor<float>(keyData, new[] { 1, _numKvHeads, kvSeqLen, _headDim });
            var valTensor = new DenseTensor<float>(valData, new[] { 1, _numKvHeads, kvSeqLen, _headDim });
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.key", keyTensor));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.value", valTensor));
        }

        using var results = _decoderSession!.Run(inputs);
        var resultList = results.ToList();

        // Extract logits (single token output)
        var logitsTensor = resultList[0].AsTensor<float>();
        var logits = new float[_vocabSize];
        CopyTensorToFlat(logitsTensor, logits);

        // Extract present KV cache
        var kvCache = new float[_numLayers * 2][];
        for (int i = 0; i < _numLayers * 2; i++)
        {
            var kv = resultList[1 + i].AsTensor<float>();
            kvCache[i] = TensorToArray(kv);
        }

        return (logits, kvCache);
    }

    private static int ArgMax(float[] data, int offset, int length)
    {
        int maxIdx = 0;
        float maxVal = data[offset];
        for (int i = 1; i < length; i++)
        {
            if (data[offset + i] > maxVal)
            {
                maxVal = data[offset + i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static void CopyTensorToFlat(Tensor<float> tensor, float[] dest)
    {
        int idx = 0;
        foreach (var val in tensor)
        {
            if (idx >= dest.Length) break;
            dest[idx++] = val;
        }
    }

    private static float[] TensorToArray(Tensor<float> tensor)
    {
        var arr = new float[tensor.Length];
        int idx = 0;
        foreach (var val in tensor)
            arr[idx++] = val;
        return arr;
    }

    private static SessionOptions CreateSessionOptions(string provider)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        if (provider == "directml")
        {
            opts.AppendExecutionProvider_DML(0);
        }

        return opts;
    }

    public void Release()
    {
        _sem.Wait();
        try { Release_Internal(); }
        finally { _sem.Release(); }
    }

    private void Release_Internal()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _encoderSession = null;
        _decoderSession = null;
        _embedTokensMatrix = null;
        _tokenizer = null;
        Debug.WriteLine("[QwenAsrOnnx] Sessions released");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Release();
        _sem.Dispose();
    }
}
