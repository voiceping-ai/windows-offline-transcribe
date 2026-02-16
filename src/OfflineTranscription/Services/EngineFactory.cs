using OfflineTranscription.Engines;
using OfflineTranscription.Interfaces;
using OfflineTranscription.Models;

namespace OfflineTranscription.Services;

/// <summary>
/// Creates the appropriate ASR engine for a model.
/// Port of iOS EngineFactory.swift.
/// </summary>
public static class EngineFactory
{
    public static IASREngine Create(ModelInfo model) => model.EngineType switch
    {
        EngineType.WhisperCpp => new WhisperCppEngine(),
        EngineType.SherpaOnnxOffline => new SherpaOnnxOfflineEngine(model.SherpaModelType),
        EngineType.SherpaOnnxStreaming => new SherpaOnnxStreamingEngine(),
        EngineType.WindowsSpeech => CreateWindowsSpeechEngine(),
        EngineType.QwenAsr => new QwenAsrEngine(),
        EngineType.QwenAsrOnnx => new QwenAsrOnnxEngine(),
        _ => throw new NotSupportedException($"Unsupported engine type: {model.EngineType}")
    };

    private static IASREngine CreateWindowsSpeechEngine()
    {
#if WINDOWS10_0_19041_0_OR_GREATER
        return new WindowsSpeechEngine();
#else
        throw new PlatformNotSupportedException(
            "WindowsSpeech engine requires Windows 10 19041+ with WinRT APIs.");
#endif
    }
}
