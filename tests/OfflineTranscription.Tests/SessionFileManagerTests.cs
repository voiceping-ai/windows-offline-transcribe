using OfflineTranscription.Utilities;

namespace OfflineTranscription.Tests;

/// <summary>
/// Tests for SessionFileManager: path normalization, relative path format,
/// back-compatibility handling, and audio existence checks.
/// </summary>
public class SessionFileManagerTests
{
    [Fact]
    public void GetAbsolutePath_NormalizesForwardSlashes()
    {
        var result = SessionFileManager.GetAbsolutePath("Sessions/abc/audio.wav");
        Assert.Contains("Sessions", result);
        Assert.Contains("abc", result);
        Assert.Contains("audio.wav", result);
        // Should not contain forward slashes on Windows (or mixed separators)
        Assert.DoesNotContain("//", result);
    }

    [Fact]
    public void GetAbsolutePath_NormalizesBackslashes()
    {
        var result = SessionFileManager.GetAbsolutePath("Sessions\\abc\\audio.wav");
        Assert.Contains("Sessions", result);
        Assert.Contains("abc", result);
    }

    [Fact]
    public void GetAbsolutePath_HandlesLowercaseSessions()
    {
        // Back-compat: older records stored "sessions/" (lowercase)
        var result = SessionFileManager.GetAbsolutePath("sessions/abc/audio.wav");
        Assert.Contains("Sessions", result);
        Assert.Contains("abc", result);
    }

    [Fact]
    public void GetAbsolutePath_StripsLeadingSlash()
    {
        var result = SessionFileManager.GetAbsolutePath("/Sessions/abc/audio.wav");
        Assert.Contains("Sessions", result);
        Assert.DoesNotContain("//", result);
    }

    [Fact]
    public void GetAbsolutePath_IsUnderLocalAppData()
    {
        var result = SessionFileManager.GetAbsolutePath("Sessions/test-id/audio.wav");
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        Assert.StartsWith(localAppData, result);
    }

    [Fact]
    public void GetAbsolutePath_ContainsOfflineTranscription()
    {
        var result = SessionFileManager.GetAbsolutePath("Sessions/test-id/audio.wav");
        Assert.Contains("OfflineTranscription", result);
    }

    [Fact]
    public void HasAudio_ReturnsFalse_ForNull()
    {
        Assert.False(SessionFileManager.HasAudio(null));
    }

    [Fact]
    public void HasAudio_ReturnsFalse_ForEmpty()
    {
        Assert.False(SessionFileManager.HasAudio(""));
    }

    [Fact]
    public void HasAudio_ReturnsFalse_ForNonexistentFile()
    {
        Assert.False(SessionFileManager.HasAudio("Sessions/nonexistent-id/audio.wav"));
    }

    [Fact]
    public void DeleteSession_DoesNotThrow_ForNonexistent()
    {
        var ex = Record.Exception(() =>
            SessionFileManager.DeleteSession("nonexistent-session-id"));
        Assert.Null(ex);
    }

    [Fact]
    public void SaveTtsEvidence_CopiesFileIntoSessionAndReturnsRelativePath()
    {
        var sessionId = $"tts-test-{Guid.NewGuid():N}";
        var sourceDir = Path.Combine(Path.GetTempPath(), $"tts_src_{Guid.NewGuid():N}");
        Directory.CreateDirectory(sourceDir);
        var sourcePath = Path.Combine(sourceDir, "source.wav");
        File.WriteAllBytes(sourcePath, new byte[] { 1, 2, 3, 4 });

        try
        {
            var rel = SessionFileManager.SaveTtsEvidence(sessionId, sourcePath);
            Assert.Equal($"Sessions/{sessionId}/tts.wav", rel);

            var abs = SessionFileManager.GetAbsolutePath(rel);
            Assert.True(File.Exists(abs));
        }
        finally
        {
            try { SessionFileManager.DeleteSession(sessionId); } catch { /* best-effort */ }
            try { Directory.Delete(sourceDir, recursive: true); } catch { /* best-effort */ }
        }
    }
}
