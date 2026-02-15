using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;
using OfflineTranscription.Models;
using OfflineTranscription.Services;
using OfflineTranscription.Views;

namespace OfflineTranscription;

public sealed partial class MainWindow : Window
{
    private bool _suppressNavSelectionChanged;
    private object? _lastNavSelection;

    public MainWindow()
    {
        this.InitializeComponent();
        SetWindowIcon();
        ContentFrame.Navigated += ContentFrame_Navigated;

        // Select mode from persisted preferences.
        _suppressNavSelectionChanged = true;
        try
        {
            NavView.SelectedItem = App.Preferences.Mode == AppMode.Translate
                ? NavTranslateItem
                : NavTranscribeItem;
        }
        finally
        {
            _suppressNavSelectionChanged = false;
        }
        _lastNavSelection = NavView.SelectedItem;

        // Navigate to model setup or transcription on launch.
        NavigateToHome();
    }

    private async void NavView_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (_suppressNavSelectionChanged) return;

        if (args.SelectedItem is NavigationViewItem item)
        {
            var tag = item.Tag?.ToString();
            switch (tag)
            {
                case "Transcribe":
                case "Translate":
                    {
                        var desiredMode = tag == "Translate" ? AppMode.Translate : AppMode.Transcribe;

                        if (App.TranscriptionService.SessionState != SessionState.Idle)
                        {
                            var dialog = new ContentDialog
                            {
                                Title = "Switch Mode",
                                Content = "Stop current recording and switch mode?",
                                PrimaryButtonText = "Stop and Switch",
                                CloseButtonText = "Cancel",
                                XamlRoot = Content.XamlRoot
                            };

                            if (await dialog.ShowAsync() != ContentDialogResult.Primary)
                            {
                                _suppressNavSelectionChanged = true;
                                try
                                {
                                    sender.SelectedItem = _lastNavSelection;
                                }
                                finally
                                {
                                    _suppressNavSelectionChanged = false;
                                }
                                return;
                            }

                            App.TranscriptionService.StopRecording();
                        }

                        App.TranscriptionService.SetMode(desiredMode);
                        NavigateToHome();
                        _lastNavSelection = sender.SelectedItem;
                        break;
                    }
                case "History":
                    ContentFrame.Navigate(typeof(HistoryPage));
                    _lastNavSelection = sender.SelectedItem;
                    break;
                case "Settings":
                    ContentFrame.Navigate(typeof(SettingsPage));
                    _lastNavSelection = sender.SelectedItem;
                    break;
            }
        }
    }

    private void NavigateToHome()
    {
        if (App.Preferences.SelectedModelId != null
            && Services.ModelDownloader.IsModelDownloaded(
                Models.ModelInfo.AvailableModels.FirstOrDefault(
                    m => m.Id == App.Preferences.SelectedModelId)
                ?? Models.ModelInfo.DefaultModel))
        {
            ContentFrame.Navigate(typeof(TranscriptionPage));
        }
        else
        {
            ContentFrame.Navigate(typeof(ModelSetupPage));
        }
    }

    private void SetWindowIcon()
    {
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        var windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(hwnd);
        var appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(windowId);
        var iconPath = Path.Combine(AppContext.BaseDirectory, "Assets", "AppIcon.ico");
        if (appWindow is not null && File.Exists(iconPath))
        {
            appWindow.SetIcon(iconPath);
        }
    }

    private void ContentFrame_Navigated(object sender, NavigationEventArgs e)
    {
        var page = e.SourcePageType?.Name ?? "Unknown";
        App.Evidence.LogEvent("navigate", new { page });
        _ = App.Evidence.CaptureScreenshotAsync($"nav_{page}");
    }
}
