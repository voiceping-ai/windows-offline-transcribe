namespace OfflineTranscription.Models;

/// <summary>
/// ASR engine backend type. Port of Android EngineType.
/// </summary>
public enum EngineType
{
    WhisperCpp,
    SherpaOnnxOffline,
    SherpaOnnxStreaming,
    WindowsSpeech,
    QwenAsr
}

/// <summary>
/// Sub-type for sherpa-onnx models. Port of Android SherpaModelType.
/// </summary>
public enum SherpaModelType
{
    Moonshine,
    SenseVoice,
    OmnilingualCtc,
    ZipformerTransducer,
    NemoTransducer
}
