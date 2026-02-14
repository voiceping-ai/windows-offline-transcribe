using System.Runtime.InteropServices;

namespace OfflineTranscription.Interop;

/// <summary>
/// P/Invoke declarations for sherpa-onnx C API.
/// sherpa-onnx.dll from GitHub releases bundles ONNX Runtime + DirectML.
/// Reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.h
/// </summary>
internal static class SherpaOnnxNative
{
    private const string LibName = "sherpa-onnx-c-api";

    // ── Offline Recognizer ──

    [DllImport(LibName, EntryPoint = "SherpaOnnxCreateOfflineRecognizer", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateOfflineRecognizer(ref SherpaOnnxOfflineRecognizerConfig config);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOfflineRecognizer", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOfflineRecognizer(IntPtr recognizer);

    [DllImport(LibName, EntryPoint = "SherpaOnnxCreateOfflineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateOfflineStream(IntPtr recognizer);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOfflineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOfflineStream(IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxAcceptWaveformOffline", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void AcceptWaveformOffline(
        IntPtr stream,
        int sampleRate,
        [In] float[] samples,
        int n);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDecodeOfflineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DecodeOfflineStream(IntPtr recognizer, IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxGetOfflineStreamResult", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr GetOfflineStreamResult(IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOfflineRecognizerResult", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOfflineRecognizerResult(IntPtr result);

    // ── Helper to read result ──

    /// <summary>
    /// Read the text from a SherpaOnnxOfflineRecognizerResult pointer.
    /// The struct layout: { const char* text; ... }
    /// </summary>
    internal static string ReadResultText(IntPtr resultPtr)
    {
        if (resultPtr == IntPtr.Zero) return "";
        // First field is char* text
        var textPtr = Marshal.ReadIntPtr(resultPtr);
        return Marshal.PtrToStringUTF8(textPtr) ?? "";
    }
}

// ── Config structs matching sherpa-onnx C API ──

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineRecognizerConfig
{
    public SherpaOnnxOfflineModelConfig ModelConfig;
    public IntPtr DecodingMethod; // "greedy_search" or "modified_beam_search"
    public int MaxActivePaths;
    public IntPtr HotwordsFile;
    public float HotwordsScore;
    public IntPtr RuleFsts;
    public IntPtr RuleFars;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineModelConfig
{
    public SherpaOnnxOfflineTransducerModelConfig Transducer;
    public SherpaOnnxOfflineParaformerModelConfig Paraformer;
    public SherpaOnnxOfflineNemoEncDecCtcModelConfig NemoCtc;
    public SherpaOnnxOfflineWhisperModelConfig Whisper;
    public SherpaOnnxOfflineTdnnModelConfig Tdnn;
    public IntPtr Tokens;
    public int NumThreads;
    [MarshalAs(UnmanagedType.I1)]
    public bool Debug;
    public IntPtr Provider; // "cpu", "directml", "cuda"
    public IntPtr ModelType;
    public IntPtr ModelingUnit;
    public IntPtr BpeVocab;
    public IntPtr TeleSpeechCtc;
    public SherpaOnnxOfflineSenseVoiceModelConfig SenseVoice;
    public SherpaOnnxOfflineMoonshineModelConfig Moonshine;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineTransducerModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
    public IntPtr Joiner;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineParaformerModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineNemoEncDecCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineWhisperModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
    public IntPtr Language;
    public IntPtr Task;
    public int TailPaddings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineTdnnModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineSenseVoiceModelConfig
{
    public IntPtr Model;
    public IntPtr Language;
    [MarshalAs(UnmanagedType.I1)]
    public bool UseInverseTextNormalization;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineMoonshineModelConfig
{
    public IntPtr Preprocessor;
    public IntPtr Encoder;
    public IntPtr UncachedDecoder;
    public IntPtr CachedDecoder;
}
