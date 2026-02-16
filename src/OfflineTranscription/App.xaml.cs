using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using OfflineTranscription.Data;
using OfflineTranscription.Interop;
using OfflineTranscription.Services;
using OfflineTranscription.Utilities;
using OfflineTranscription.ViewModels;

namespace OfflineTranscription;

public partial class App : Application
{
    public static Window? MainWindow { get; private set; }

    // ── Shared services (poor man's DI) ──
    public static AppPreferences Preferences { get; private set; } = null!;
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
        // Best-effort migration from the legacy OfflineSpeechTranslation Windows app.
        LegacyMigration.TryMigrateFromOfflineSpeechTranslation();
        Preferences = new AppPreferences();

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

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern int AddDllDirectory(string newDirectory);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetDefaultDllDirectories(uint directoryFlags);

    /// <summary>
    /// Configure DLL import resolver for native libraries (whisper.dll, sherpa-onnx).
    /// Native DLLs are placed alongside the exe in the output directory.
    /// </summary>
    private static void SetupNativeLibraryResolvers()
    {
        var basePath = AppContext.BaseDirectory;

        // Pre-load dependency DLLs so the Windows loader can find them.
        // ggml: whisper.dll imports ggml.dll and ggml-base.dll.
        // libopenblas: qwen_asr.dll links against OpenBLAS.
        // libstdc++/libgomp/libctranslate2: translation engine dependencies.
        foreach (var dep in new[] { "ggml-base", "ggml-cpu", "ggml", "libopenblas", "libstdc++-6", "libgomp-1", "libctranslate2" })
        {
            var depPath = Path.Combine(basePath, $"{dep}.dll");
            if (File.Exists(depPath))
            {
                try { NativeLibrary.Load(depPath); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[App] Failed to pre-load {dep}.dll: {ex.Message}");
                }
            }
        }

        NativeLibrary.SetDllImportResolver(typeof(WhisperNative).Assembly, (name, assembly, searchPath) =>
        {
            // Check runtimes/win-x64/ subfolder
            var runtimePath = Path.Combine(basePath, "runtimes", "win-x64", $"{name}.dll");
            if (File.Exists(runtimePath))
            {
                try { return NativeLibrary.Load(runtimePath); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[App] Failed to load {runtimePath}: {ex.Message}");
                }
            }

            // Check app base directory directly
            var directPath = Path.Combine(basePath, $"{name}.dll");
            if (File.Exists(directPath))
            {
                try { return NativeLibrary.Load(directPath); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[App] Failed to load {directPath}: {ex.Message}");
                }
            }

            // Fallback to default resolution
            return IntPtr.Zero;
        });
    }
}
