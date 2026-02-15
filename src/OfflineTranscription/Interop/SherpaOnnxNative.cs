using System.Runtime.InteropServices;

namespace OfflineTranscription.Interop;

/// <summary>
/// P/Invoke declarations for sherpa-onnx C API.
/// sherpa-onnx.dll from GitHub releases bundles ONNX Runtime + DirectML.
/// Reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.h
/// IMPORTANT: Struct layouts must exactly match the native C API header.
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

    // ── Online (Streaming) Recognizer ──

    [DllImport(LibName, EntryPoint = "SherpaOnnxCreateOnlineRecognizer", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateOnlineRecognizer(ref SherpaOnnxOnlineRecognizerConfig config);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOnlineRecognizer", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOnlineRecognizer(IntPtr recognizer);

    [DllImport(LibName, EntryPoint = "SherpaOnnxCreateOnlineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateOnlineStream(IntPtr recognizer);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOnlineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOnlineStream(IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxOnlineStreamAcceptWaveform", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void OnlineStreamAcceptWaveform(
        IntPtr stream,
        int sampleRate,
        [In] float[] samples,
        int n);

    [DllImport(LibName, EntryPoint = "SherpaOnnxIsOnlineStreamReady", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int IsOnlineStreamReady(IntPtr recognizer, IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDecodeOnlineStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DecodeOnlineStream(IntPtr recognizer, IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxGetOnlineStreamResultAsJson", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr GetOnlineStreamResultAsJson(IntPtr recognizer, IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxDestroyOnlineStreamResultJson", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void DestroyOnlineStreamResultJson(IntPtr json);

    [DllImport(LibName, EntryPoint = "SherpaOnnxOnlineStreamReset", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void OnlineStreamReset(IntPtr recognizer, IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxOnlineStreamInputFinished", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void OnlineStreamInputFinished(IntPtr stream);

    [DllImport(LibName, EntryPoint = "SherpaOnnxOnlineStreamIsEndpoint", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int OnlineStreamIsEndpoint(IntPtr recognizer, IntPtr stream);

    /// <summary>
    /// Read text from an Online result JSON string pointer.
    /// </summary>
    internal static string ReadOnlineResultJson(IntPtr jsonPtr)
    {
        if (jsonPtr == IntPtr.Zero) return "";
        return Marshal.PtrToStringUTF8(jsonPtr) ?? "";
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Offline config structs — must match sherpa-onnx c-api.h exactly
// ══════════════════════════════════════════════════════════════════════════════

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxFeatureConfig
{
    public int SampleRate;
    public int FeatureDim;
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
    public int EnableTokenTimestamps;
    public int EnableSegmentTimestamps;
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
    public int UseItn; // int32_t, NOT bool
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineMoonshineModelConfig
{
    public IntPtr Preprocessor;
    public IntPtr Encoder;
    public IntPtr UncachedDecoder;
    public IntPtr CachedDecoder;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineFireRedAsrModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineDolphinModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineZipformerCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineCanaryModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
    public IntPtr SrcLang;
    public IntPtr TgtLang;
    public int UsePnc;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineWenetCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineOmnilingualAsrCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineMedAsrCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineFunASRNanoModelConfig
{
    public IntPtr EncoderAdaptor;
    public IntPtr Llm;
    public IntPtr Embedding;
    public IntPtr Tokenizer;
    public IntPtr SystemPrompt;
    public IntPtr UserPrompt;
    public int MaxNewTokens;
    public float Temperature;
    public float TopP;
    public int Seed;
    public IntPtr Language;
    public int Itn;
    public IntPtr Hotwords;
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
    public int Debug; // int32_t, NOT bool
    public IntPtr Provider; // "cpu", "directml", "cuda"
    public IntPtr ModelType;
    public IntPtr ModelingUnit;
    public IntPtr BpeVocab;
    public IntPtr TeleSpeechCtc;
    public SherpaOnnxOfflineSenseVoiceModelConfig SenseVoice;
    public SherpaOnnxOfflineMoonshineModelConfig Moonshine;
    public SherpaOnnxOfflineFireRedAsrModelConfig FireRedAsr;
    public SherpaOnnxOfflineDolphinModelConfig Dolphin;
    public SherpaOnnxOfflineZipformerCtcModelConfig ZipformerCtc;
    public SherpaOnnxOfflineCanaryModelConfig Canary;
    public SherpaOnnxOfflineWenetCtcModelConfig WenetCtc;
    public SherpaOnnxOfflineOmnilingualAsrCtcModelConfig Omnilingual;
    public SherpaOnnxOfflineMedAsrCtcModelConfig MedAsr;
    public SherpaOnnxOfflineFunASRNanoModelConfig FunasrNano;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineLMConfig
{
    public IntPtr Model;
    public float Scale;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxHomophoneReplacerConfig
{
    public IntPtr DictDir;
    public IntPtr Lexicon;
    public IntPtr RuleFsts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOfflineRecognizerConfig
{
    public SherpaOnnxFeatureConfig FeatConfig;
    public SherpaOnnxOfflineModelConfig ModelConfig;
    public SherpaOnnxOfflineLMConfig LmConfig;
    public IntPtr DecodingMethod; // "greedy_search" or "modified_beam_search"
    public int MaxActivePaths;
    public IntPtr HotwordsFile;
    public float HotwordsScore;
    public IntPtr RuleFsts;
    public IntPtr RuleFars;
    public float BlankPenalty;
    public SherpaOnnxHomophoneReplacerConfig Hr;
}

// ══════════════════════════════════════════════════════════════════════════════
// Online (Streaming) config structs — must match sherpa-onnx c-api.h exactly
// ══════════════════════════════════════════════════════════════════════════════

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineTransducerModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
    public IntPtr Joiner;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineParaformerModelConfig
{
    public IntPtr Encoder;
    public IntPtr Decoder;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineZipformer2CtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineNemoCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineToneCtcModelConfig
{
    public IntPtr Model;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineModelConfig
{
    public SherpaOnnxOnlineTransducerModelConfig Transducer;
    public SherpaOnnxOnlineParaformerModelConfig Paraformer;
    public SherpaOnnxOnlineZipformer2CtcModelConfig Zipformer2Ctc;
    public IntPtr Tokens;
    public int NumThreads;
    public IntPtr Provider;
    public int Debug;
    public IntPtr ModelType;
    public IntPtr ModelingUnit;
    public IntPtr BpeVocab;
    public IntPtr TokensBuf;
    public int TokensBufSize;
    public SherpaOnnxOnlineNemoCtcModelConfig NemoCtc;
    public SherpaOnnxOnlineToneCtcModelConfig TOneCtc;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineCtcFstDecoderConfig
{
    public IntPtr Graph;
    public int MaxActive;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SherpaOnnxOnlineRecognizerConfig
{
    public SherpaOnnxFeatureConfig FeatConfig;
    public SherpaOnnxOnlineModelConfig ModelConfig;
    public IntPtr DecodingMethod;
    public int MaxActivePaths;
    public int EnableEndpoint;
    public float Rule1MinTrailingSilence;
    public float Rule2MinTrailingSilence;
    public float Rule3MinUtteranceLength;
    public IntPtr HotwordsFile;
    public float HotwordsScore;
    public SherpaOnnxOnlineCtcFstDecoderConfig CtcFstDecoderConfig;
    public IntPtr RuleFsts;
    public IntPtr RuleFars;
    public float BlankPenalty;
    public IntPtr HotwordsBuf;
    public int HotwordsBufSize;
    public SherpaOnnxHomophoneReplacerConfig Hr;
}
