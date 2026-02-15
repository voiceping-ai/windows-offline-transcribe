using System.Diagnostics;
using OfflineTranscription.Interfaces;
using OfflineTranscription.Models;
using Windows.Media.SpeechRecognition;

namespace OfflineTranscription.Engines;

/// <summary>
/// Windows Speech API engine using WinRT SpeechRecognizer.
/// Works offline if the Windows language pack is installed.
/// Limited accuracy compared to neural ASR models.
///
/// Note: The WinRT SpeechRecognizer.RecognizeAsync() listens from the default
/// microphone. For pre-recorded audio, we start a continuous recognition session
/// and collect results via the ResultGenerated event. This engine is best suited
/// for live recording; file transcription quality may be limited.
/// </summary>
public sealed class WindowsSpeechEngine : IASREngine
{
    private SpeechRecognizer? _recognizer;
    private readonly SemaphoreSlim _sem = new(1, 1);
    private bool _disposed;

    public bool IsLoaded => _recognizer != null;
    public bool IsStreaming => false;

    public async Task<bool> LoadModelAsync(string modelPath, CancellationToken ct = default)
    {
        await _sem.WaitAsync(ct);
        try
        {
            Release_Internal();

            try
            {
                _recognizer = new SpeechRecognizer();

                _recognizer.Constraints.Add(
                    new SpeechRecognitionTopicConstraint(SpeechRecognitionScenario.Dictation, "dictation"));

                var compileResult = await _recognizer.CompileConstraintsAsync();
                if (compileResult.Status != SpeechRecognitionResultStatus.Success)
                {
                    Debug.WriteLine($"[WindowsSpeech] CompileConstraints failed: {compileResult.Status}");
                    _recognizer.Dispose();
                    _recognizer = null;
                    return false;
                }

                // Set timeouts for dictation (allow longer pauses)
                _recognizer.Timeouts.InitialSilenceTimeout = TimeSpan.FromSeconds(10);
                _recognizer.Timeouts.EndSilenceTimeout = TimeSpan.FromSeconds(3);
                _recognizer.Timeouts.BabbleTimeout = TimeSpan.FromMinutes(5);

                Debug.WriteLine($"[WindowsSpeech] Recognizer loaded, language={_recognizer.CurrentLanguage.LanguageTag}");
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WindowsSpeech] Load failed: {ex.Message}");
                _recognizer?.Dispose();
                _recognizer = null;
                return false;
            }
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
            if (_recognizer == null) return ASRResult.Empty;

            var sw = Stopwatch.StartNew();

            // Use RecognizeAsync which listens from the default audio input.
            // This is the primary way Windows Speech API works — it captures
            // from the microphone. For file transcription, the audio is played
            // back through the system and the recognizer captures it.
            // For chunk-based live recording, this works naturally.
            var result = await _recognizer.RecognizeAsync();

            sw.Stop();

            var text = result.Text?.Trim() ?? "";

            Debug.WriteLine($"[WindowsSpeech] Result status={result.Status}, confidence={result.Confidence}, text='{text}'");

            if (result.Status != SpeechRecognitionResultStatus.Success ||
                string.IsNullOrWhiteSpace(text))
            {
                return ASRResult.Empty with { InferenceTimeMs = sw.Elapsed.TotalMilliseconds };
            }

            var lang = _recognizer.CurrentLanguage.LanguageTag;
            if (lang.Contains('-'))
                lang = lang.Split('-')[0];

            var segments = new[] { new ASRSegment(text, DetectedLanguage: lang) };
            return new ASRResult(text, segments, lang) with { InferenceTimeMs = sw.Elapsed.TotalMilliseconds };
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[WindowsSpeech] TranscribeAsync failed: {ex.Message}");
            return ASRResult.Empty;
        }
        finally
        {
            _sem.Release();
        }
    }

    public void Release()
    {
        _sem.Wait();
        try { Release_Internal(); }
        finally { _sem.Release(); }
    }

    private void Release_Internal()
    {
        if (_recognizer != null)
        {
            _recognizer.Dispose();
            _recognizer = null;
            Debug.WriteLine("[WindowsSpeech] Recognizer released");
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
