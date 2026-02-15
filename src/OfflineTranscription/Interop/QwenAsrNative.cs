using System.Runtime.InteropServices;

namespace OfflineTranscription.Interop;

/// <summary>
/// P/Invoke declarations for qwen-asr C library (antirez/qwen-asr).
/// qwen_asr.dll must be compiled from source and placed in libs/runtimes/win-x64/.
/// Reference: https://github.com/antirez/qwen-asr
/// </summary>
internal static class QwenAsrNative
{
    private const string LibName = "qwen_asr";

    /// <summary>Load a model from the given directory. Returns context pointer or null on failure.</summary>
    [DllImport(LibName, EntryPoint = "qwen_load", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern IntPtr Load([MarshalAs(UnmanagedType.LPUTF8Str)] string modelDir);

    /// <summary>Free a loaded model context.</summary>
    [DllImport(LibName, EntryPoint = "qwen_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void Free(IntPtr ctx);

    /// <summary>
    /// Transcribe raw float audio samples (16 kHz mono).
    /// Returns a malloc'd UTF-8 string that must be freed with FreeString.
    /// </summary>
    [DllImport(LibName, EntryPoint = "qwen_transcribe_audio", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr TranscribeAudio(IntPtr ctx, [In] float[] samples, int numSamples);

    /// <summary>
    /// Optionally force a language for transcription.
    /// Returns 0 on success, non-zero on error.
    /// </summary>
    [DllImport(LibName, EntryPoint = "qwen_set_force_language", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern int SetForceLanguage(IntPtr ctx, [MarshalAs(UnmanagedType.LPUTF8Str)] string language);

    /// <summary>Free a string returned by qwen_transcribe_audio (uses the DLL's own free).</summary>
    [DllImport(LibName, EntryPoint = "qwen_free_string", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeString(IntPtr str);

    /// <summary>
    /// Read a UTF-8 string from a native pointer and free the native memory.
    /// qwen-asr returns malloc'd strings that must be freed via the DLL's own free().
    /// </summary>
    internal static string ReadAndFreeString(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return "";
        var text = Marshal.PtrToStringUTF8(ptr) ?? "";
        FreeString(ptr);
        return text;
    }
}
