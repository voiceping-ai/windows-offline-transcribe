using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.IO;
using NAudio.Wave.SampleProviders;
using OfflineTranscription.Models;
using OfflineTranscription.Services;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage.Pickers;

namespace OfflineTranscription.ViewModels;

/// <summary>
/// ViewModel for the transcription page. Wraps TranscriptionService
/// and exposes commands for the UI.
/// </summary>
public sealed partial class TranscriptionViewModel : ObservableObject
{
    private readonly TranscriptionService _service;
    private readonly AppPreferences _prefs;

    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private string _currentModelName = "No model loaded";
    [ObservableProperty] private string _inferenceLabel = "";

    public TranscriptionService Service => _service;

    public TranscriptionViewModel(TranscriptionService service, AppPreferences prefs)
    {
        _service = service;
        _prefs = prefs;

        // Keep UI state in sync with service state.
        IsRecording = _service.SessionState == SessionState.Recording;
        _service.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(TranscriptionService.SessionState))
                IsRecording = _service.SessionState == SessionState.Recording;
        };
    }

    [RelayCommand]
    private void ToggleRecording()
    {
        if (_service.SessionState == SessionState.Recording)
        {
            App.Evidence.LogEvent("ui_record_toggle", new { action = "stop" });
            _service.StopRecording();
        }
        else if (_service.ModelState == ASRModelState.Loaded)
        {
            App.Evidence.LogEvent("ui_record_toggle", new { action = "start", source = _prefs.CaptureSource.ToString() });
            _service.StartRecording(_prefs.CaptureSource);
        }
    }

    [RelayCommand]
    private async Task TranscribeFileAsync()
    {
        if (_service.ModelState != ASRModelState.Loaded) return;

        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".wav");
        picker.FileTypeFilter.Add(".mp3");
        picker.SuggestedStartLocation = PickerLocationId.MusicLibrary;

        // WinUI 3 picker needs window handle
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindow);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSingleFileAsync();
        if (file == null) return;

        try
        {
            var info = new FileInfo(file.Path);
            App.Evidence.LogEvent("file_transcribe_selected", new
            {
                name = Path.GetFileName(file.Path),
                sizeBytes = info.Exists ? info.Length : (long?)null
            });
            _ = App.Evidence.CaptureScreenshotAsync("file_transcribe_selected");
        }
        catch
        {
            App.Evidence.LogEvent("file_transcribe_selected", new { name = Path.GetFileName(file.Path) });
        }

        // Decode audio file to 16kHz mono float samples.
        var samples = await ReadAudioFileAsync(file.Path);
        if (samples != null)
        {
            var result = await _service.TranscribeFileAsync(samples);

            // Log evidence with or without full transcript text based on preference.
            if (_prefs.EvidenceIncludeTranscriptText)
            {
                App.Evidence.LogEvent("file_transcribe_done", new
                {
                    text = result.Text,
                    detectedLanguage = result.DetectedLanguage,
                    inferenceTimeMs = result.InferenceTimeMs
                });
            }
            else
            {
                App.Evidence.LogEvent("file_transcribe_done", new
                {
                    textLength = result.Text?.Length ?? 0,
                    detectedLanguage = result.DetectedLanguage,
                    inferenceTimeMs = result.InferenceTimeMs
                });
            }
            _ = App.Evidence.CaptureScreenshotAsync("file_transcribe_done");
        }
    }

    [RelayCommand]
    private void CopyText()
    {
        var text = $"{_service.ConfirmedText} {_service.HypothesisText}".Trim();
        if (string.IsNullOrEmpty(text)) return;

        var dp = new DataPackage();
        dp.SetText(text);
        Clipboard.SetContent(dp);
    }

    [RelayCommand]
    private void ClearTranscription()
    {
        _service.ClearTranscription();
    }

    public async Task LoadSavedModelAsync()
    {
        var modelId = _prefs.SelectedModelId;
        if (modelId == null) return;

        var model = ModelInfo.AvailableModels.FirstOrDefault(m => m.Id == modelId);
        if (model != null && ModelDownloader.IsModelDownloaded(model))
        {
            await _service.SelectAndLoadModelAsync(model);
            CurrentModelName = model.DisplayName;
            InferenceLabel = model.InferenceMethod;
        }
    }

    /// <summary>Read an audio file (wav/mp3/...) into float[] samples at 16kHz mono.</summary>
    private static async Task<float[]?> ReadAudioFileAsync(string path)
    {
        return await Task.Run(() =>
        {
            try
            {
                using var reader = new NAudio.Wave.AudioFileReader(path);
                NAudio.Wave.ISampleProvider provider = reader;

                // Convert to mono if needed
                if (provider.WaveFormat.Channels > 1)
                    provider = provider.ToMono();

                // Resample if needed
                if (provider.WaveFormat.SampleRate != 16000)
                    provider = new NAudio.Wave.SampleProviders.WdlResamplingSampleProvider(provider, 16000);

                var samples = new List<float>();
                var buffer = new float[4096];
                int read;
                while ((read = provider.Read(buffer, 0, buffer.Length)) > 0)
                {
                    for (int i = 0; i < read; i++)
                        samples.Add(buffer[i]);
                }
                return samples.ToArray();
            }
            catch
            {
                return null;
            }
        });
    }
}
