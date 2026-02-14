using System.Diagnostics;
using System.Runtime.InteropServices;
using OfflineTranscription.Interfaces;
using OfflineTranscription.Interop;
using OfflineTranscription.Models;

namespace OfflineTranscription.Engines;

/// <summary>
/// whisper.cpp engine via P/Invoke. Port of Android WhisperCppEngine.kt.
/// Thread-safe: all native calls guarded by SemaphoreSlim.
/// </summary>
public sealed class WhisperCppEngine : IASREngine
{
    private IntPtr _ctx = IntPtr.Zero;
    private readonly SemaphoreSlim _sem = new(1, 1);
    private bool _disposed;

    public bool IsLoaded => _ctx != IntPtr.Zero;
    public bool IsStreaming => false;

    public async Task<bool> LoadModelAsync(string modelPath, CancellationToken ct = default)
    {
        await _sem.WaitAsync(ct);
        try
        {
            // Release previous context if any
            if (_ctx != IntPtr.Zero)
            {
                WhisperNative.Free(_ctx);
                _ctx = IntPtr.Zero;
            }

            // Load on thread pool to avoid blocking UI
            var ctx = await Task.Run(() =>
            {
                var cparams = WhisperNative.ContextDefaultParams();
                cparams.use_gpu = false; // CPU only for now; GPU via DirectML in sherpa-onnx
                return WhisperNative.InitFromFile(modelPath, cparams);
            }, ct);

            if (ctx == IntPtr.Zero)
            {
                Debug.WriteLine($"[WhisperCpp] Failed to load model: {modelPath}");
                return false;
            }

            _ctx = ctx;
            Debug.WriteLine($"[WhisperCpp] Model loaded: {modelPath}");
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
        if (!IsLoaded)
            return ASRResult.Empty;

        await _sem.WaitAsync(ct);
        try
        {
            var ctx = _ctx;
            if (ctx == IntPtr.Zero) return ASRResult.Empty;

            var sw = Stopwatch.StartNew();

            var result = await Task.Run(() =>
            {
                var wparams = WhisperNative.FullDefaultParams(WhisperNative.WHISPER_SAMPLING_GREEDY);
                wparams.n_threads = numThreads;
                wparams.print_progress = false;
                wparams.print_special = false;
                wparams.print_realtime = false;
                wparams.print_timestamps = false;
                wparams.no_timestamps = false;
                wparams.single_segment = false;
                wparams.translate = false;
                wparams.suppress_blank = true;

                // Language handling: whisper.cpp treats language detection as a separate flag.
                // If caller asks for "auto", enable detect_language and pass a safe default.
                bool detectLanguage = string.IsNullOrWhiteSpace(language) ||
                                      language.Equals("auto", StringComparison.OrdinalIgnoreCase);
                wparams.detect_language = detectLanguage;

                var langToPass = detectLanguage ? "en" : language;
                var langPtr = Marshal.StringToCoTaskMemUTF8(langToPass);
                wparams.language = langPtr;

                try
                {
                    int ret = WhisperNative.Full(ctx, wparams, audioSamples, audioSamples.Length);
                    if (ret != 0)
                    {
                        Debug.WriteLine($"[WhisperCpp] whisper_full returned {ret}");
                        return ASRResult.Empty;
                    }

                    // Read detected language from the native API
                    string? detectedLang = null;
                    if (detectLanguage)
                    {
                        try
                        {
                            int langId = WhisperNative.FullLangId(ctx);
                            if (langId >= 0)
                            {
                                var langStrPtr = WhisperNative.LangStr(langId);
                                detectedLang = Marshal.PtrToStringUTF8(langStrPtr);
                            }
                        }
                        catch { /* best-effort language detection */ }
                    }
                    else
                    {
                        detectedLang = language;
                    }

                    int nSegments = WhisperNative.FullNSegments(ctx);
                    var segments = new List<ASRSegment>(nSegments);
                    var fullText = new System.Text.StringBuilder();

                    for (int i = 0; i < nSegments; i++)
                    {
                        var textPtr = WhisperNative.FullGetSegmentText(ctx, i);
                        var text = Marshal.PtrToStringUTF8(textPtr) ?? "";
                        var t0 = WhisperNative.FullGetSegmentT0(ctx, i) * 10; // centiseconds → ms
                        var t1 = WhisperNative.FullGetSegmentT1(ctx, i) * 10;

                        segments.Add(new ASRSegment(text.Trim(), t0, t1));
                        fullText.Append(text);
                    }

                    return new ASRResult(
                        fullText.ToString().Trim(),
                        segments,
                        DetectedLanguage: detectedLang);
                }
                finally
                {
                    Marshal.FreeCoTaskMem(langPtr);
                }
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
                WhisperNative.Free(_ctx);
                _ctx = IntPtr.Zero;
                Debug.WriteLine("[WhisperCpp] Context released");
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
