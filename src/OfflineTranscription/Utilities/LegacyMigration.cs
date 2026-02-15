using System.Diagnostics;
using System.Text.Json;
using OfflineTranscription.Models;

namespace OfflineTranscription.Utilities;

public static class LegacyMigration
{
    public static void TryMigrateFromOfflineSpeechTranslation()
    {
        try
        {
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var oldBaseDir = Path.Combine(localAppData, "OfflineSpeechTranslation");
            var newBaseDir = Path.Combine(localAppData, "OfflineTranscription");
            var markerPath = Path.Combine(newBaseDir, ".migrated_from_offline_speech_translation");

            if (File.Exists(markerPath)) return;

            // Nothing to migrate.
            if (!Directory.Exists(oldBaseDir) && !File.Exists(Path.Combine(oldBaseDir, "settings.json")))
                return;

            Directory.CreateDirectory(newBaseDir);

            // Settings.json needs schema mapping, do it first.
            TryMigrateSettings(
                oldSettingsPath: Path.Combine(oldBaseDir, "settings.json"),
                newSettingsPath: Path.Combine(newBaseDir, "settings.json"));

            // DB + model/session directories
            TryMoveFileOrCopyIfNeeded(
                sourcePath: Path.Combine(oldBaseDir, "transcriptions.db"),
                destPath: Path.Combine(newBaseDir, "transcriptions.db"));

            TryMoveDirectoryOrCopyIfNeeded(
                sourceDir: Path.Combine(oldBaseDir, "Models"),
                destDir: Path.Combine(newBaseDir, "Models"));

            TryMoveDirectoryOrCopyIfNeeded(
                sourceDir: Path.Combine(oldBaseDir, "TranslationModels"),
                destDir: Path.Combine(newBaseDir, "TranslationModels"));

            TryMoveDirectoryOrCopyIfNeeded(
                sourceDir: Path.Combine(oldBaseDir, "Sessions"),
                destDir: Path.Combine(newBaseDir, "Sessions"));

            TryMoveDirectoryOrCopyIfNeeded(
                sourceDir: Path.Combine(oldBaseDir, "TtsEvidence"),
                destDir: Path.Combine(newBaseDir, "TtsEvidence"));

            File.WriteAllText(markerPath, DateTime.UtcNow.ToString("O"));
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[LegacyMigration] Migration failed: {ex.Message}");
        }
    }

    private static void TryMigrateSettings(string oldSettingsPath, string newSettingsPath)
    {
        try
        {
            if (!File.Exists(oldSettingsPath)) return;
            if (File.Exists(newSettingsPath)) return;

            var json = File.ReadAllText(oldSettingsPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            bool translationEnabled = TryGetBool(root, "TranslationEnabled") ?? false;
            var mode = translationEnabled ? AppMode.Translate : AppMode.Transcribe;

            var newSettings = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase)
            {
                ["SelectedModelId"] = TryGetString(root, "SelectedModelId"),
                ["UseVAD"] = TryGetBool(root, "UseVAD") ?? true,
                ["ShowTimestamps"] = TryGetBool(root, "ShowTimestamps") ?? false,
                ["CaptureSource"] = TryGetString(root, "CaptureSource") ?? "Microphone",
                ["EvidenceMode"] = TryGetBool(root, "EvidenceMode") ?? false,
                ["EvidenceSessionFolder"] = TryGetString(root, "EvidenceSessionFolder"),
                ["EvidenceIncludeTranscriptText"] = TryGetBool(root, "EvidenceIncludeTranscriptText") ?? false,

                ["Mode"] = mode.ToString(),
                ["TranslationSourceLanguageCode"] = TryGetString(root, "TranslationSourceLanguageCode") ?? "en",
                ["TranslationTargetLanguageCode"] = TryGetString(root, "TranslationTargetLanguageCode") ?? "ja",
                ["SpeakTranslatedAudio"] = TryGetBool(root, "SpeakTranslatedAudio") ?? true,
                ["TtsRate"] = TryGetNumber(root, "TtsRate") ?? 1.0,
                ["TtsVoiceId"] = TryGetString(root, "TtsVoiceId")
            };

            var outJson = JsonSerializer.Serialize(newSettings, new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(Path.GetDirectoryName(newSettingsPath)!);
            File.WriteAllText(newSettingsPath, outJson);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[LegacyMigration] Settings migration failed: {ex.Message}");
        }
    }

    private static string? TryGetString(JsonElement root, string name)
    {
        try
        {
            if (!root.TryGetProperty(name, out var el)) return null;
            if (el.ValueKind == JsonValueKind.String) return el.GetString();
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static bool? TryGetBool(JsonElement root, string name)
    {
        try
        {
            if (!root.TryGetProperty(name, out var el)) return null;
            return el.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.String => bool.TryParse(el.GetString(), out var b) ? b : null,
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private static double? TryGetNumber(JsonElement root, string name)
    {
        try
        {
            if (!root.TryGetProperty(name, out var el)) return null;
            if (el.ValueKind == JsonValueKind.Number && el.TryGetDouble(out var n)) return n;
            if (el.ValueKind == JsonValueKind.String && double.TryParse(el.GetString(), out n)) return n;
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static void TryMoveFileOrCopyIfNeeded(string sourcePath, string destPath)
    {
        try
        {
            if (!File.Exists(sourcePath)) return;
            if (File.Exists(destPath)) return;

            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

            try
            {
                File.Move(sourcePath, destPath);
            }
            catch
            {
                File.Copy(sourcePath, destPath);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[LegacyMigration] File migrate failed: {Path.GetFileName(sourcePath)}: {ex.Message}");
        }
    }

    private static void TryMoveDirectoryOrCopyIfNeeded(string sourceDir, string destDir)
    {
        try
        {
            if (!Directory.Exists(sourceDir)) return;
            if (Directory.Exists(destDir)) return;

            Directory.CreateDirectory(Path.GetDirectoryName(destDir)!);

            try
            {
                Directory.Move(sourceDir, destDir);
            }
            catch
            {
                CopyDirectoryRecursive(sourceDir, destDir);
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[LegacyMigration] Dir migrate failed: {Path.GetFileName(sourceDir)}: {ex.Message}");
        }
    }

    private static void CopyDirectoryRecursive(string sourceDir, string destDir)
    {
        Directory.CreateDirectory(destDir);

        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var dest = Path.Combine(destDir, Path.GetFileName(file));
            File.Copy(file, dest, overwrite: false);
        }

        foreach (var dir in Directory.GetDirectories(sourceDir))
        {
            var dest = Path.Combine(destDir, Path.GetFileName(dir));
            CopyDirectoryRecursive(dir, dest);
        }
    }
}

