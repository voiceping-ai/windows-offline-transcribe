using System.Diagnostics;
using System.Runtime.InteropServices;
using OfflineTranscription.Interfaces;
using OfflineTranscription.Interop;
using OfflineTranscription.Models;

namespace OfflineTranscription.Engines;

/// <summary>
/// qwen-asr engine via P/Invoke (antirez/qwen-asr).
/// Supports Qwen3-ASR models with BF16 safetensors.
/// Thread-safe: all native calls guarded by SemaphoreSlim.
/// </summary>
public sealed class QwenAsrEngine : IASREngine
{
    private IntPtr _ctx = IntPtr.Zero;
    private readonly SemaphoreSlim _sem = new(1, 1);
    private bool _disposed;

    public bool IsLoaded => _ctx != IntPtr.Zero;
    public bool IsStreaming => false;

    public async Task<bool> LoadModelAsync(string modelDir, CancellationToken ct = default)
    {
        await _sem.WaitAsync(ct);
        try
        {
            if (_ctx != IntPtr.Zero)
            {
                QwenAsrNative.Free(_ctx);
                _ctx = IntPtr.Zero;
            }

            var ctx = await Task.Run(() =>
            {
                Debug.WriteLine($"[QwenAsr] Loading model from {modelDir}");
                return QwenAsrNative.Load(modelDir);
            }, ct);

            if (ctx == IntPtr.Zero)
            {
                Debug.WriteLine("[QwenAsr] Failed to load model");
                return false;
            }

            _ctx = ctx;
            Debug.WriteLine("[QwenAsr] Model loaded successfully");
            return true;
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
            var ctx = _ctx;
            if (ctx == IntPtr.Zero) return ASRResult.Empty;

            var sw = Stopwatch.StartNew();

            var result = await Task.Run(() =>
            {
                // Set language if specified (non-auto)
                if (!string.IsNullOrEmpty(language) && language != "auto")
                {
                    QwenAsrNative.SetForceLanguage(ctx, language);
                }

                Debug.WriteLine($"[QwenAsr] Transcribing {audioSamples.Length} samples ({audioSamples.Length / 16000.0:F1}s)");

                var textPtr = QwenAsrNative.TranscribeAudio(ctx, audioSamples, audioSamples.Length);
                var text = QwenAsrNative.ReadAndFreeString(textPtr);

                text = text.Trim();
                Debug.WriteLine($"[QwenAsr] Result: '{text}'");

                if (string.IsNullOrWhiteSpace(text))
                    return ASRResult.Empty;

                // Qwen3-ASR supports 52 languages but doesn't report detected language
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

    public void Release()
    {
        _sem.Wait();
        try
        {
            if (_ctx != IntPtr.Zero)
            {
                QwenAsrNative.Free(_ctx);
                _ctx = IntPtr.Zero;
                Debug.WriteLine("[QwenAsr] Context released");
            }
        }
        finally
        {
            _sem.Release();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Release();
        _sem.Dispose();
    }
}
