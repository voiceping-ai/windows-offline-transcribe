using OfflineTranscription.Models;

namespace OfflineTranscription.Tests;

public class ModelInfoTests
{
    [Fact]
    public void AvailableModels_HasThreeModels()
    {
        Assert.Equal(3, ModelInfo.AvailableModels.Count);
    }

    [Fact]
    public void DefaultModel_IsWhisperBase()
    {
        Assert.Equal("whisper-base", ModelInfo.DefaultModel.Id);
    }

    [Fact]
    public void AllModels_HaveFiles()
    {
        foreach (var model in ModelInfo.AvailableModels)
        {
            Assert.NotEmpty(model.Files);
        }
    }

    [Fact]
    public void AllModels_HaveValidUrls()
    {
        foreach (var model in ModelInfo.AvailableModels)
        {
            foreach (var file in model.Files)
            {
                Assert.True(Uri.IsWellFormedUriString(file.Url, UriKind.Absolute),
                    $"Invalid URL for {model.Id}: {file.Url}");
            }
        }
    }

    [Fact]
    public void AllModels_HaveUniqueIds()
    {
        var ids = ModelInfo.AvailableModels.Select(m => m.Id).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Fact]
    public void ModelsByEngine_GroupsCorrectly()
    {
        var byEngine = ModelInfo.ModelsByEngine;

        Assert.True(byEngine.ContainsKey(EngineType.WhisperCpp));
        Assert.True(byEngine.ContainsKey(EngineType.SherpaOnnxOffline));

        Assert.Single(byEngine[EngineType.WhisperCpp]);           // whisper-base
        Assert.Equal(2, byEngine[EngineType.SherpaOnnxOffline].Count); // sensevoice + moonshine
    }

    [Fact]
    public void WhisperModel_HasSingleFile()
    {
        var whisper = ModelInfo.AvailableModels.First(m => m.Id == "whisper-base");
        Assert.Single(whisper.Files);
        Assert.Contains("ggml-base.bin", whisper.Files[0].LocalName);
    }

    [Fact]
    public void MoonshineModel_HasFiveFiles()
    {
        var moonshine = ModelInfo.AvailableModels.First(m => m.Id == "moonshine-tiny");
        Assert.Equal(5, moonshine.Files.Count);
        Assert.Contains(moonshine.Files, f => f.LocalName == "tokens.txt");
        Assert.Contains(moonshine.Files, f => f.LocalName == "preprocess.onnx");
    }

    [Fact]
    public void SenseVoiceModel_HasTwoFiles()
    {
        var sv = ModelInfo.AvailableModels.First(m => m.Id == "sensevoice-small");
        Assert.Equal(2, sv.Files.Count);
        Assert.Equal(SherpaModelType.SenseVoice, sv.SherpaModelType);
    }

    [Fact]
    public void InferenceMethod_ReturnsCorrectStrings()
    {
        var whisper = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.WhisperCpp);
        Assert.Contains("whisper.cpp", whisper.InferenceMethod);

        var sherpa = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.SherpaOnnxOffline);
        Assert.Contains("sherpa-onnx", sherpa.InferenceMethod);
    }
}
