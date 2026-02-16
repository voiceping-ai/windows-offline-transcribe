namespace OfflineTranscription.Models;

/// <summary>
/// Metadata for an available ASR model. Port of Android ModelInfo.kt.
/// 13 models across 6 engine types.
/// </summary>
public record ModelInfo(
    string Id,
    string DisplayName,
    EngineType EngineType,
    string ParameterCount,
    string SizeOnDisk,
    string Description,
    string Languages,
    IReadOnlyList<ModelFile> Files,
    SherpaModelType? SherpaModelType = null
)
{
    public string InferenceMethod => EngineType switch
    {
        EngineType.WhisperCpp => "whisper.cpp (C++/P-Invoke)",
        EngineType.SherpaOnnxOffline => "sherpa-onnx offline (ONNX Runtime)",
        EngineType.SherpaOnnxStreaming => "sherpa-onnx streaming (ONNX Runtime)",
        EngineType.WindowsSpeech => "Windows Speech API (WinRT)",
        EngineType.QwenAsr => "qwen-asr (C/P-Invoke)",
        EngineType.QwenAsrOnnx => "qwen-asr (ONNX Runtime/DirectML)",
        _ => "Unknown"
    };

    // ── HuggingFace base URLs (matching Android ModelInfo.kt) ──

    private const string WhisperBaseUrl =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

    private const string MoonshineTinyBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-tiny-en-int8/resolve/main/";

    private const string MoonshineBaseBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/";

    private const string SenseVoiceBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/";

    private const string OmnilingualBaseUrl =
        "https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/resolve/main/";

    private const string ZipformerEnBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/";

    private const string ParakeetV2BaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/";

    private const string Qwen3Asr06BBaseUrl =
        "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/";

    private const string Qwen3Asr06BOnnxBaseUrl =
        "https://huggingface.co/voiceping-ai/qwen3-asr-0.6b-onnx/resolve/main/";

    private static IReadOnlyList<ModelFile> MoonshineFiles(string baseUrl) =>
    [
        new($"{baseUrl}preprocess.onnx", "preprocess.onnx"),
        new($"{baseUrl}encode.int8.onnx", "encode.int8.onnx"),
        new($"{baseUrl}uncached_decode.int8.onnx", "uncached_decode.int8.onnx"),
        new($"{baseUrl}cached_decode.int8.onnx", "cached_decode.int8.onnx"),
        new($"{baseUrl}tokens.txt", "tokens.txt"),
    ];

    /// <summary>
    /// All available models (13 models across 6 engine types).
    /// </summary>
    public static IReadOnlyList<ModelInfo> AvailableModels { get; } =
    [
        // ── Whisper (whisper.cpp) ──
        new(
            Id: "whisper-tiny",
            DisplayName: "Whisper Tiny",
            EngineType: EngineType.WhisperCpp,
            ParameterCount: "39M",
            SizeOnDisk: "~80 MB",
            Description: "Fastest Whisper model. Good for quick notes.",
            Languages: "99 languages",
            Files: [new($"{WhisperBaseUrl}ggml-tiny.bin", "ggml-tiny.bin")]
        ),
        new(
            Id: "whisper-base",
            DisplayName: "Whisper Base",
            EngineType: EngineType.WhisperCpp,
            ParameterCount: "74M",
            SizeOnDisk: "~150 MB",
            Description: "Balanced speed and accuracy. Recommended for general use.",
            Languages: "99 languages",
            Files: [new($"{WhisperBaseUrl}ggml-base.bin", "ggml-base.bin")]
        ),
        new(
            Id: "whisper-small",
            DisplayName: "Whisper Small",
            EngineType: EngineType.WhisperCpp,
            ParameterCount: "244M",
            SizeOnDisk: "~500 MB",
            Description: "Higher accuracy, slower. Best for important recordings.",
            Languages: "99 languages",
            Files: [new($"{WhisperBaseUrl}ggml-small.bin", "ggml-small.bin")]
        ),
        new(
            Id: "whisper-large-v3-turbo",
            DisplayName: "Whisper Large V3 Turbo",
            EngineType: EngineType.WhisperCpp,
            ParameterCount: "809M",
            SizeOnDisk: "~834 MB",
            Description: "Near-SOTA accuracy. Best quality, larger download.",
            Languages: "99 languages",
            Files: [new($"{WhisperBaseUrl}ggml-large-v3-turbo-q8_0.bin", "ggml-large-v3-turbo-q8_0.bin")]
        ),

        // ── SenseVoice (sherpa-onnx offline) ──
        new(
            Id: "sensevoice-small",
            DisplayName: "SenseVoice Small",
            EngineType: EngineType.SherpaOnnxOffline,
            ParameterCount: "234M",
            SizeOnDisk: "~240 MB",
            Description: "Multilingual (zh/en/ja/ko/yue). 5x faster than Whisper Small.",
            Languages: "zh/en/ja/ko/yue",
            Files:
            [
                new($"{SenseVoiceBaseUrl}model.int8.onnx", "model.int8.onnx"),
                new($"{SenseVoiceBaseUrl}tokens.txt", "tokens.txt"),
            ],
            SherpaModelType: Models.SherpaModelType.SenseVoice
        ),

        // ── Moonshine (sherpa-onnx offline) ──
        new(
            Id: "moonshine-tiny",
            DisplayName: "Moonshine Tiny",
            EngineType: EngineType.SherpaOnnxOffline,
            ParameterCount: "27M",
            SizeOnDisk: "~125 MB",
            Description: "Ultra-fast, English only. 5x faster than Whisper Tiny.",
            Languages: "English",
            Files: MoonshineFiles(MoonshineTinyBaseUrl),
            SherpaModelType: Models.SherpaModelType.Moonshine
        ),
        new(
            Id: "moonshine-base",
            DisplayName: "Moonshine Base",
            EngineType: EngineType.SherpaOnnxOffline,
            ParameterCount: "61M",
            SizeOnDisk: "~290 MB",
            Description: "Fast English with higher accuracy than Tiny.",
            Languages: "English",
            Files: MoonshineFiles(MoonshineBaseBaseUrl),
            SherpaModelType: Models.SherpaModelType.Moonshine
        ),

        // ── Omnilingual (sherpa-onnx offline, NemoCtc) ──
        new(
            Id: "omnilingual-300m",
            DisplayName: "Omnilingual 300M",
            EngineType: EngineType.SherpaOnnxOffline,
            ParameterCount: "300M",
            SizeOnDisk: "~365 MB",
            Description: "1,600+ languages. Facebook MMS CTC model, int8 quantized.",
            Languages: "1,600+ languages",
            Files:
            [
                new($"{OmnilingualBaseUrl}model.int8.onnx", "model.int8.onnx"),
                new($"{OmnilingualBaseUrl}tokens.txt", "tokens.txt"),
            ],
            SherpaModelType: Models.SherpaModelType.OmnilingualCtc
        ),

        // ── Parakeet TDT v2 (sherpa-onnx offline, NemoTransducer) ──
        new(
            Id: "parakeet-tdt-v2",
            DisplayName: "Parakeet TDT v2",
            EngineType: EngineType.SherpaOnnxOffline,
            ParameterCount: "600M",
            SizeOnDisk: "~660 MB",
            Description: "NVIDIA Parakeet TDT v2. Fast English transducer, int8 quantized.",
            Languages: "English",
            Files:
            [
                new($"{ParakeetV2BaseUrl}encoder.int8.onnx", "encoder.int8.onnx"),
                new($"{ParakeetV2BaseUrl}decoder.int8.onnx", "decoder.int8.onnx"),
                new($"{ParakeetV2BaseUrl}joiner.int8.onnx", "joiner.int8.onnx"),
                new($"{ParakeetV2BaseUrl}tokens.txt", "tokens.txt"),
            ],
            SherpaModelType: Models.SherpaModelType.NemoTransducer
        ),

        // ── Zipformer Streaming (sherpa-onnx streaming) ──
        new(
            Id: "zipformer-20m",
            DisplayName: "Zipformer Streaming",
            EngineType: EngineType.SherpaOnnxStreaming,
            ParameterCount: "20M",
            SizeOnDisk: "~73 MB",
            Description: "Real-time streaming English. Ultra-low latency.",
            Languages: "English",
            Files:
            [
                new($"{ZipformerEnBaseUrl}encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
                new($"{ZipformerEnBaseUrl}decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
                new($"{ZipformerEnBaseUrl}joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
                new($"{ZipformerEnBaseUrl}tokens.txt", "tokens.txt"),
            ],
            SherpaModelType: Models.SherpaModelType.ZipformerTransducer
        ),
        // ── Qwen3-ASR (qwen-asr C engine) ──
        new(
            Id: "qwen3-asr-0.6b",
            DisplayName: "Qwen3 ASR 0.6B",
            EngineType: EngineType.QwenAsr,
            ParameterCount: "600M",
            SizeOnDisk: "~1.9 GB",
            Description: "Qwen3 ASR 0.6B. 52 languages, BF16 safetensors via qwen-asr C engine.",
            Languages: "52 languages",
            Files:
            [
                new($"{Qwen3Asr06BBaseUrl}model.safetensors", "model.safetensors"),
                new($"{Qwen3Asr06BBaseUrl}vocab.json", "vocab.json"),
                new($"{Qwen3Asr06BBaseUrl}merges.txt", "merges.txt"),
                new($"{Qwen3Asr06BBaseUrl}config.json", "config.json"),
                new($"{Qwen3Asr06BBaseUrl}generation_config.json", "generation_config.json"),
            ]
        ),

        // ── Qwen3-ASR ONNX (ONNX Runtime / DirectML) ──
        new(
            Id: "qwen3-asr-0.6b-onnx",
            DisplayName: "Qwen3 ASR 0.6B (ONNX)",
            EngineType: EngineType.QwenAsrOnnx,
            ParameterCount: "600M",
            SizeOnDisk: "~1.2 GB",
            Description: "Qwen3 ASR 0.6B via ONNX Runtime. DirectML GPU acceleration with CPU fallback.",
            Languages: "52 languages",
            Files:
            [
                new($"{Qwen3Asr06BOnnxBaseUrl}encoder.onnx", "encoder.onnx"),
                new($"{Qwen3Asr06BOnnxBaseUrl}decoder.onnx", "decoder.onnx"),
                new($"{Qwen3Asr06BOnnxBaseUrl}embed_tokens.bin", "embed_tokens.bin"),
                new($"{Qwen3Asr06BOnnxBaseUrl}vocab.json", "vocab.json"),
                new($"{Qwen3Asr06BOnnxBaseUrl}merges.txt", "merges.txt"),
            ]
        ),

        // ── Windows Speech API (built-in) ──
        new(
            Id: "windows-speech",
            DisplayName: "Windows Speech (Built-in)",
            EngineType: EngineType.WindowsSpeech,
            ParameterCount: "N/A",
            SizeOnDisk: "0 MB (pre-installed)",
            Description: "Windows built-in recognizer. Limited accuracy. Requires language pack install.",
            Languages: "Depends on installed language packs",
            Files: []
        ),
    ];

    public static ModelInfo DefaultModel => AvailableModels.First(m => m.Id == "whisper-base");

    /// <summary>
    /// Group models by engine type for UI display (computed once, not per access).
    /// </summary>
    public static IReadOnlyDictionary<EngineType, IReadOnlyList<ModelInfo>> ModelsByEngine { get; } =
        AvailableModels
            .GroupBy(m => m.EngineType)
            .ToDictionary(g => g.Key, g => (IReadOnlyList<ModelInfo>)g.ToList());
}
