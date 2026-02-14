using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using OfflineTranscription.Data;
using OfflineTranscription.Interop;
using OfflineTranscription.Services;
using OfflineTranscription.ViewModels;

namespace OfflineTranscription;

public partial class App : Application
{
    public static Window? MainWindow { get; private set; }

    // ── Shared services (poor man's DI) ──
    public static AppPreferences Preferences { get; } = new();
    public static EvidenceService Evidence { get; private set; } = null!;
    public static TranscriptionService TranscriptionService { get; private set; } = null!;
    public static TranscriptionViewModel TranscriptionVM { get; private set; } = null!;

    public App()
    {
        this.InitializeComponent();
        SetupNativeLibraryResolvers();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        Evidence = new EvidenceService(Preferences);
        TranscriptionService = new TranscriptionService(Preferences);
        TranscriptionVM = new TranscriptionViewModel(TranscriptionService, Preferences);
        Evidence.AttachTranscriptionService(TranscriptionService);

        try
        {
            AppDbContext.EnsureCreated();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[App] Failed to initialize database: {ex.Message}");
        }

        MainWindow = new MainWindow();
        MainWindow.Activate();

        // Start evidence capture if enabled (real-device diagnostics).
        Evidence.TryAutoStart();
        _ = Evidence.CaptureScreenshotAsync("app_launch");
    }

    /// <summary>
    /// Configure DLL import resolver for native libraries (whisper.dll, sherpa-onnx).
    /// Native DLLs are placed alongside the exe in the output directory.
    /// </summary>
    private static void SetupNativeLibraryResolvers()
    {
        NativeLibrary.SetDllImportResolver(typeof(WhisperNative).Assembly, (name, assembly, searchPath) =>
        {
            // Try loading from app base directory first
            var basePath = AppContext.BaseDirectory;

            // Check runtimes/win-x64/ subfolder
            var runtimePath = Path.Combine(basePath, "runtimes", "win-x64", $"{name}.dll");
            if (File.Exists(runtimePath))
                return NativeLibrary.Load(runtimePath);

            // Check app base directory directly
            var directPath = Path.Combine(basePath, $"{name}.dll");
            if (File.Exists(directPath))
                return NativeLibrary.Load(directPath);

            // Fallback to default resolution
            return IntPtr.Zero;
        });
    }
}
