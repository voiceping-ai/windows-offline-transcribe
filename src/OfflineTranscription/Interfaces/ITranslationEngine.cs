namespace OfflineTranscription.Interfaces;

public interface ITranslationEngine : IDisposable
{
    bool ModelReady { get; }
    double DownloadProgress { get; }
    string? DownloadStatus { get; }
    string? Warning { get; }

    Task PrepareAsync(string sourceLanguageCode, string targetLanguageCode, CancellationToken ct = default);

    Task<string> TranslateAsync(
        string text,
        string sourceLanguageCode,
        string targetLanguageCode,
        CancellationToken ct = default);
}

