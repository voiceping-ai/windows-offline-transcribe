using System.Diagnostics;
using OfflineTranscription.Models;

namespace OfflineTranscription.Services;

/// <summary>
/// Downloads model files from HuggingFace with progress and resume.
/// Port of Android ModelDownloader.kt + iOS ModelDownloader.swift.
/// </summary>
public sealed class ModelDownloader
{
    private static readonly HttpClient _client = new()
    {
        Timeout = TimeSpan.FromMinutes(10)
    };

    /// <summary>
    /// Base directory for all downloaded models.
    /// %LOCALAPPDATA%\OfflineTranscription\Models\
    /// </summary>
    public static string ModelsBaseDir =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "OfflineTranscription", "Models");

    /// <summary>
    /// Get the model directory for a specific model.
    /// </summary>
    public static string GetModelDir(ModelInfo model) =>
        Path.Combine(ModelsBaseDir, model.Id);

    /// <summary>
    /// Check if all files for a model are downloaded.
    /// </summary>
    public static bool IsModelDownloaded(ModelInfo model)
    {
        var dir = GetModelDir(model);
        return model.Files.All(f => File.Exists(Path.Combine(dir, f.LocalName)));
    }

    /// <summary>
    /// Get the primary model file path (for engine loading).
    /// For whisper: returns the .bin file. For sherpa-onnx: returns the directory.
    /// </summary>
    public static string GetModelPath(ModelInfo model)
    {
        var dir = GetModelDir(model);
        return model.EngineType == EngineType.WhisperCpp
            ? Path.Combine(dir, model.Files[0].LocalName)
            : dir;
    }

    /// <summary>
    /// Download all files for a model with progress reporting.
    /// Supports resume via HTTP Range headers.
    /// </summary>
    public static async Task DownloadAsync(
        ModelInfo model,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        var dir = GetModelDir(model);
        Directory.CreateDirectory(dir);

        int totalFiles = model.Files.Count;

        for (int i = 0; i < totalFiles; i++)
        {
            var file = model.Files[i];
            var targetPath = Path.Combine(dir, file.LocalName);
            var tempPath = targetPath + ".tmp";

            // Skip if already downloaded
            if (File.Exists(targetPath))
            {
                progress?.Report((i + 1.0) / totalFiles);
                continue;
            }

            long existingBytes = 0;
            if (File.Exists(tempPath))
                existingBytes = new FileInfo(tempPath).Length;

            using var request = new HttpRequestMessage(HttpMethod.Get, file.Url);

            // Resume support
            if (existingBytes > 0)
            {
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existingBytes, null);
                Debug.WriteLine($"[ModelDownloader] Resuming {file.LocalName} from byte {existingBytes}");
            }

            using var response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);

            // If server doesn't support Range, start fresh
            if (existingBytes > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent)
            {
                existingBytes = 0;
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }

            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            if (existingBytes > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent)
                totalBytes += existingBytes;

            await using var contentStream = await response.Content.ReadAsStreamAsync(ct);
            await using var fileStream = new FileStream(tempPath,
                existingBytes > 0 ? FileMode.Append : FileMode.Create,
                FileAccess.Write, FileShare.None, 81920, true);

            var buffer = new byte[81920];
            long bytesWritten = existingBytes;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
                bytesWritten += bytesRead;

                if (totalBytes > 0)
                {
                    double fileProgress = (double)bytesWritten / totalBytes;
                    double overallProgress = (i + fileProgress) / totalFiles;
                    progress?.Report(overallProgress);
                }
            }

            await fileStream.FlushAsync(ct);

            // Size verification before rename (Android M2 fix)
            if (totalBytes > 0 && bytesWritten != totalBytes)
            {
                File.Delete(tempPath);
                throw new IOException(
                    $"Download incomplete for {file.LocalName}: expected {totalBytes} bytes, got {bytesWritten}");
            }

            // Atomic temp → final rename
            File.Move(tempPath, targetPath, overwrite: true);
            Debug.WriteLine($"[ModelDownloader] Downloaded {file.LocalName} ({bytesWritten:N0} bytes)");

            progress?.Report((i + 1.0) / totalFiles);
        }
    }

    /// <summary>
    /// Delete all files for a model to reclaim disk space.
    /// </summary>
    public static void DeleteModel(ModelInfo model)
    {
        var dir = GetModelDir(model);
        if (Directory.Exists(dir))
        {
            Directory.Delete(dir, recursive: true);
            Debug.WriteLine($"[ModelDownloader] Deleted model: {model.Id}");
        }
    }
}
