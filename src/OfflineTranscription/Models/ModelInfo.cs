namespace OfflineTranscription.Models;

/// <summary>
/// Metadata for an available ASR model. Port of Android ModelInfo.kt.
/// v1 ships 3 core models: whisper-base, sensevoice-small, moonshine-tiny.
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
        _ => "Unknown"
    };

    // ── HuggingFace base URLs (matching Android ModelInfo.kt) ──

    private const string WhisperBaseUrl =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

    private const string MoonshineTinyBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-tiny-en-int8/resolve/main/";

    private const string SenseVoiceBaseUrl =
        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/";

    private static IReadOnlyList<ModelFile> MoonshineFiles(string baseUrl) =>
    [
        new($"{baseUrl}preprocess.onnx", "preprocess.onnx"),
        new($"{baseUrl}encode.int8.onnx", "encode.int8.onnx"),
        new($"{baseUrl}uncached_decode.int8.onnx", "uncached_decode.int8.onnx"),
        new($"{baseUrl}cached_decode.int8.onnx", "cached_decode.int8.onnx"),
        new($"{baseUrl}tokens.txt", "tokens.txt"),
    ];

    /// <summary>
    /// All available models for v1 (3 core models).
    /// </summary>
    public static IReadOnlyList<ModelInfo> AvailableModels { get; } =
    [
        // ── Whisper (whisper.cpp) ──
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

        // ── SenseVoice (sherpa-onnx) ──
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

        // ── Moonshine Tiny (sherpa-onnx) ──
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
    ];

    public static ModelInfo DefaultModel => AvailableModels[0]; // whisper-base

    /// <summary>
    /// Group models by engine type for UI display (computed once, not per access).
    /// </summary>
    public static IReadOnlyDictionary<EngineType, IReadOnlyList<ModelInfo>> ModelsByEngine { get; } =
        AvailableModels
            .GroupBy(m => m.EngineType)
            .ToDictionary(g => g.Key, g => (IReadOnlyList<ModelInfo>)g.ToList());
}
