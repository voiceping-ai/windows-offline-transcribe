using OfflineTranscription.Models;

namespace OfflineTranscription.Tests;

/// <summary>
/// Tests for TranscriptionRecord: default values, unique IDs, nullable fields.
/// </summary>
public class TranscriptionRecordTests
{
    [Fact]
    public void DefaultId_IsNonEmpty()
    {
        var record = new TranscriptionRecord();
        Assert.False(string.IsNullOrEmpty(record.Id));
    }

    [Fact]
    public void DefaultId_IsValidGuid()
    {
        var record = new TranscriptionRecord();
        Assert.True(Guid.TryParse(record.Id, out _));
    }

    [Fact]
    public void TwoRecords_HaveDifferentIds()
    {
        var r1 = new TranscriptionRecord();
        var r2 = new TranscriptionRecord();
        Assert.NotEqual(r1.Id, r2.Id);
    }

    [Fact]
    public void DefaultText_IsEmpty()
    {
        var record = new TranscriptionRecord();
        Assert.Equal("", record.Text);
    }

    [Fact]
    public void DefaultModelUsed_IsEmpty()
    {
        var record = new TranscriptionRecord();
        Assert.Equal("", record.ModelUsed);
    }

    [Fact]
    public void DefaultDuration_IsZero()
    {
        var record = new TranscriptionRecord();
        Assert.Equal(0.0, record.DurationSeconds);
    }

    [Fact]
    public void Language_IsNullByDefault()
    {
        var record = new TranscriptionRecord();
        Assert.Null(record.Language);
    }

    [Fact]
    public void AudioFileName_IsNullByDefault()
    {
        var record = new TranscriptionRecord();
        Assert.Null(record.AudioFileName);
    }

    [Fact]
    public void CreatedAt_IsRecentUtc()
    {
        var before = DateTime.UtcNow;
        var record = new TranscriptionRecord();
        var after = DateTime.UtcNow;

        Assert.InRange(record.CreatedAt, before, after);
    }

    [Fact]
    public void AllProperties_AreSettable()
    {
        var record = new TranscriptionRecord
        {
            Id = "custom-id",
            Text = "Test text",
            TranslatedText = "Translated text",
            CreatedAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc),
            DurationSeconds = 42.5,
            ModelUsed = "whisper-base",
            Language = "en",
            TranslationSourceLanguage = "en",
            TranslationTargetLanguage = "ja",
            TranslationModelId = "ct2-opus-mt-en-ja-int8",
            AudioFileName = "Sessions/abc/audio.wav",
            TtsEvidenceFileName = "Sessions/abc/tts.wav"
        };

        Assert.Equal("custom-id", record.Id);
        Assert.Equal("Test text", record.Text);
        Assert.Equal("Translated text", record.TranslatedText);
        Assert.Equal(42.5, record.DurationSeconds);
        Assert.Equal("whisper-base", record.ModelUsed);
        Assert.Equal("en", record.Language);
        Assert.Equal("en", record.TranslationSourceLanguage);
        Assert.Equal("ja", record.TranslationTargetLanguage);
        Assert.Equal("ct2-opus-mt-en-ja-int8", record.TranslationModelId);
        Assert.Equal("Sessions/abc/audio.wav", record.AudioFileName);
        Assert.Equal("Sessions/abc/tts.wav", record.TtsEvidenceFileName);
    }
}
