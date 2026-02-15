namespace OfflineTranscription.Models;

/// <summary>
/// Audio capture source selected by the user.
/// Stored in preferences and used by the recorder.
/// </summary>
public enum CaptureSource
{
    Microphone,
    SystemLoopback
}

