using System.Text;
using System.Text.Json;

namespace OfflineTranscription.Tokenizers;

/// <summary>
/// Minimal BPE decoder for Qwen3-ASR. Decode-only (no encode needed since
/// we use fixed token IDs for prompts). Matches python_simple_implementation.py.
/// </summary>
internal sealed class QwenTokenizer
{
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<char, byte> _byteDecoder;

    // Special token IDs (from tokenizer_config.json / generation_config.json)
    public const int TokenImStart = 151644;
    public const int TokenImEnd = 151645;
    public const int TokenAudioStart = 151669;
    public const int TokenAudioEnd = 151670;
    public const int TokenAudioPad = 151676;
    public const int TokenEndOfText = 151643;
    public const int TokenAsrText = 151704;

    /// <summary>EOS token IDs that stop generation.</summary>
    public static readonly HashSet<int> EosTokenIds = new() { TokenEndOfText, TokenImEnd };

    /// <summary>
    /// Prompt prefix: &lt;|im_start|&gt;system\n&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n&lt;|audio_start|&gt;
    /// </summary>
    public static readonly int[] PromptPrefix =
        { TokenImStart, 8948, 198, TokenImEnd, 198, TokenImStart, 872, 198, TokenAudioStart };

    /// <summary>
    /// Prompt suffix: &lt;|audio_end|&gt;&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n
    /// </summary>
    public static readonly int[] PromptSuffix =
        { TokenAudioEnd, TokenImEnd, 198, TokenImStart, 77091, 198 };

    // Token IDs >= this threshold are special tokens
    private const int SpecialTokenThreshold = 151643;

    public QwenTokenizer(string vocabPath)
    {
        // Load vocab.json: maps token_string -> token_id
        var json = File.ReadAllText(vocabPath, Encoding.UTF8);
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(json)!;

        _idToToken = new Dictionary<int, string>(vocab.Count);
        foreach (var (token, id) in vocab)
            _idToToken[id] = token;

        // Build byte decoder (reverse of GPT-2 byte encoding)
        _byteDecoder = new Dictionary<char, byte>();
        var byteEncoder = BytesToUnicode();
        foreach (var (b, c) in byteEncoder)
            _byteDecoder[c] = b;
    }

    /// <summary>
    /// Decode a list of token IDs to text.
    /// Strips special tokens, handles byte-level BPE encoding.
    /// </summary>
    public string Decode(List<int> tokenIds)
    {
        var pieces = new StringBuilder();
        foreach (var tid in tokenIds)
        {
            // Skip special tokens
            if (tid >= SpecialTokenThreshold)
                continue;

            if (_idToToken.TryGetValue(tid, out var tokenStr))
                pieces.Append(tokenStr);
        }

        // Decode byte-level BPE: each character maps to a byte via GPT-2 encoding
        var text = pieces.ToString();
        var bytes = new List<byte>(text.Length);
        foreach (var c in text)
        {
            if (_byteDecoder.TryGetValue(c, out var b))
                bytes.Add(b);
        }

        var result = Encoding.UTF8.GetString(bytes.ToArray());

        // Parse ASR output: "language <lang><asr_text>transcription"
        // The <asr_text> token (151704) is stripped as special, but the text
        // before it contains language info we don't need
        return result;
    }

    /// <summary>
    /// GPT-2 style byte-to-unicode mapping used by Qwen2 tokenizer.
    /// </summary>
    private static Dictionary<byte, char> BytesToUnicode()
    {
        var bs = new List<int>();
        // Printable ASCII range
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        // Latin-1 supplement ranges
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var result = new Dictionary<byte, char>(256);
        for (int i = 0; i < bs.Count; i++)
            result[(byte)bs[i]] = (char)cs[i];

        return result;
    }
}
