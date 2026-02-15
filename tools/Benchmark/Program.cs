using System.Diagnostics;
using NAudio.Wave;
using OfflineTranscription.Engines;
using OfflineTranscription.Models;
using OfflineTranscription.Services;

// ── Parse arguments ──
string? modelId = args.Length > 0 ? args[0] : null;
string audioPath = args.Length > 1 ? args[1] : FindTestAudio();

if (modelId == null)
{
    Console.WriteLine("Usage: Benchmark <model-id> [audio-file]");
    Console.WriteLine();
    Console.WriteLine("Available models:");
    foreach (var m in ModelInfo.AvailableModels)
    {
        if (m.EngineType == EngineType.WindowsSpeech) continue;
        var downloaded = ModelDownloader.IsModelDownloaded(m) ? "[downloaded]" : "[not downloaded]";
        Console.WriteLine($"  {m.Id,-25} {m.EngineType,-20} {downloaded}");
    }
    return;
}

var model = ModelInfo.AvailableModels.FirstOrDefault(m => m.Id == modelId);
if (model == null)
{
    Console.Error.WriteLine($"Unknown model: {modelId}");
    return;
}

if (!ModelDownloader.IsModelDownloaded(model))
{
    Console.Error.WriteLine($"Model {modelId} is not downloaded. Download it first via the app.");
    return;
}

Console.WriteLine($"Model:    {model.Id} ({model.EngineType})");
Console.WriteLine($"Audio:    {audioPath}");
Console.WriteLine();

// ── Load audio ──
float[] samples = LoadWav(audioPath);
double audioDurationSec = samples.Length / 16000.0;
Console.WriteLine($"Audio:    {audioDurationSec:F1}s ({samples.Length} samples at 16kHz)");

// ── Create and load engine ──
using var engine = EngineFactory.Create(model);
string modelPath = ModelDownloader.GetModelPath(model);

Console.Write("Loading model... ");
var loadSw = Stopwatch.StartNew();
bool loaded = await engine.LoadModelAsync(modelPath);
loadSw.Stop();

if (!loaded)
{
    Console.Error.WriteLine("FAILED to load model.");
    return;
}
Console.WriteLine($"done in {loadSw.ElapsedMilliseconds}ms");

// ── Transcribe ──
Console.Write("Transcribing... ");
var result = await engine.TranscribeAsync(samples, 4, "en");
Console.WriteLine($"done in {result.InferenceTimeMs:F0}ms");

// ── Results ──
double rtf = result.InferenceTimeMs / 1000.0 / audioDurationSec;
int wordCount = string.IsNullOrWhiteSpace(result.Text)
    ? 0
    : result.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
double tokPerSec = wordCount > 0 && result.InferenceTimeMs > 0
    ? wordCount / (result.InferenceTimeMs / 1000.0)
    : 0;

Console.WriteLine();
Console.WriteLine("── Results ──");
Console.WriteLine($"Text:           {result.Text}");
Console.WriteLine($"Language:       {result.DetectedLanguage ?? "N/A"}");
Console.WriteLine($"Inference:      {result.InferenceTimeMs:F0}ms");
Console.WriteLine($"RTF:            {rtf:F3}x (lower is faster)");
Console.WriteLine($"Words:          {wordCount}");
Console.WriteLine($"Words/sec:      {tokPerSec:F1}");
Console.WriteLine($"Audio duration: {audioDurationSec:F1}s");

if (engine is SherpaOnnxOfflineEngine sherpa)
    Console.WriteLine($"Provider:       {sherpa.Provider}");

// ── Helper functions ──

static string FindTestAudio()
{
    var candidates = new[]
    {
        Path.Combine(AppContext.BaseDirectory, "test-english-30s.wav"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "test-english-30s.wav"),
        @"C:\Users\voice\windows-offline-transcribe\test-english-30s.wav"
    };
    return candidates.FirstOrDefault(File.Exists)
        ?? throw new FileNotFoundException("test-english-30s.wav not found");
}

static float[] LoadWav(string path)
{
    using var reader = new AudioFileReader(path);

    // Resample to 16kHz mono if needed
    var format = reader.WaveFormat;
    var allSamples = new List<float>();
    var buffer = new float[format.SampleRate * format.Channels];
    int read;
    while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
    {
        for (int i = 0; i < read; i += format.Channels)
            allSamples.Add(buffer[i]); // mono: take first channel
    }

    if (format.SampleRate != 16000)
    {
        // Simple linear resampling
        double ratio = 16000.0 / format.SampleRate;
        int newLen = (int)(allSamples.Count * ratio);
        var resampled = new float[newLen];
        for (int i = 0; i < newLen; i++)
        {
            double srcIdx = i / ratio;
            int idx0 = (int)srcIdx;
            int idx1 = Math.Min(idx0 + 1, allSamples.Count - 1);
            double frac = srcIdx - idx0;
            resampled[i] = (float)(allSamples[idx0] * (1 - frac) + allSamples[idx1] * frac);
        }
        return resampled;
    }

    return allSamples.ToArray();
}
