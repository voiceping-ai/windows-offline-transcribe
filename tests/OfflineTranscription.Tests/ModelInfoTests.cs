using OfflineTranscription.Models;

namespace OfflineTranscription.Tests;

public class ModelInfoTests
{
    [Fact]
    public void AvailableModels_HasExpectedCount()
    {
        // 4 whisper + 2 moonshine + 1 sensevoice + 1 omnilingual + 1 parakeet + 1 zipformer + 1 qwen3 + 1 qwen3-onnx + 1 windows-speech = 13
        Assert.Equal(13, ModelInfo.AvailableModels.Count);
    }

    [Fact]
    public void DefaultModel_IsWhisperBase()
    {
        Assert.Equal("whisper-base", ModelInfo.DefaultModel.Id);
    }

    [Fact]
    public void AllModels_HaveFiles_ExceptWindowsSpeech()
    {
        foreach (var model in ModelInfo.AvailableModels)
        {
            if (model.EngineType == EngineType.WindowsSpeech)
            {
                Assert.Empty(model.Files);
            }
            else
            {
                Assert.NotEmpty(model.Files);
            }
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
        Assert.True(byEngine.ContainsKey(EngineType.SherpaOnnxStreaming));
        Assert.True(byEngine.ContainsKey(EngineType.QwenAsr));
        Assert.True(byEngine.ContainsKey(EngineType.WindowsSpeech));

        Assert.Equal(4, byEngine[EngineType.WhisperCpp].Count);         // tiny, base, small, large-v3-turbo
        Assert.Equal(5, byEngine[EngineType.SherpaOnnxOffline].Count);  // sensevoice, moonshine-tiny, moonshine-base, omnilingual, parakeet-tdt-v2
        Assert.Single(byEngine[EngineType.SherpaOnnxStreaming]);         // zipformer-20m
        Assert.Single(byEngine[EngineType.QwenAsr]);                    // qwen3-asr-0.6b
        Assert.True(byEngine.ContainsKey(EngineType.QwenAsrOnnx));
        Assert.Single(byEngine[EngineType.QwenAsrOnnx]);               // qwen3-asr-0.6b-onnx
        Assert.Single(byEngine[EngineType.WindowsSpeech]);              // windows-speech
    }

    [Fact]
    public void WhisperModel_HasSingleFile()
    {
        var whisper = ModelInfo.AvailableModels.First(m => m.Id == "whisper-base");
        Assert.Single(whisper.Files);
        Assert.Contains("ggml-base.bin", whisper.Files[0].LocalName);
    }

    [Fact]
    public void WhisperTiny_HasCorrectFile()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "whisper-tiny");
        Assert.Single(model.Files);
        Assert.Contains("ggml-tiny.bin", model.Files[0].LocalName);
    }

    [Fact]
    public void WhisperSmall_HasCorrectFile()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "whisper-small");
        Assert.Single(model.Files);
        Assert.Contains("ggml-small.bin", model.Files[0].LocalName);
    }

    [Fact]
    public void WhisperLargeV3Turbo_HasCorrectFile()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "whisper-large-v3-turbo");
        Assert.Single(model.Files);
        Assert.Contains("ggml-large-v3-turbo-q8_0.bin", model.Files[0].LocalName);
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
    public void MoonshineBase_HasFiveFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "moonshine-base");
        Assert.Equal(5, model.Files.Count);
        Assert.Equal(SherpaModelType.Moonshine, model.SherpaModelType);
    }

    [Fact]
    public void SenseVoiceModel_HasTwoFiles()
    {
        var sv = ModelInfo.AvailableModels.First(m => m.Id == "sensevoice-small");
        Assert.Equal(2, sv.Files.Count);
        Assert.Equal(SherpaModelType.SenseVoice, sv.SherpaModelType);
    }

    [Fact]
    public void OmnilingualModel_HasTwoFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "omnilingual-300m");
        Assert.Equal(2, model.Files.Count);
        Assert.Equal(SherpaModelType.OmnilingualCtc, model.SherpaModelType);
        Assert.Equal(EngineType.SherpaOnnxOffline, model.EngineType);
    }

    [Fact]
    public void ZipformerModel_HasFourFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "zipformer-20m");
        Assert.Equal(4, model.Files.Count);
        Assert.Equal(SherpaModelType.ZipformerTransducer, model.SherpaModelType);
        Assert.Equal(EngineType.SherpaOnnxStreaming, model.EngineType);
        Assert.Contains(model.Files, f => f.LocalName == "tokens.txt");
        Assert.Contains(model.Files, f => f.LocalName.Contains("encoder"));
    }

    [Fact]
    public void ParakeetTdtV2_HasFourFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "parakeet-tdt-v2");
        Assert.Equal(4, model.Files.Count);
        Assert.Equal(SherpaModelType.NemoTransducer, model.SherpaModelType);
        Assert.Equal(EngineType.SherpaOnnxOffline, model.EngineType);
        Assert.Contains(model.Files, f => f.LocalName == "tokens.txt");
        Assert.Contains(model.Files, f => f.LocalName.Contains("encoder"));
        Assert.Contains(model.Files, f => f.LocalName.Contains("decoder"));
        Assert.Contains(model.Files, f => f.LocalName.Contains("joiner"));
    }

    [Fact]
    public void Qwen3Asr_HasFiveFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "qwen3-asr-0.6b");
        Assert.Equal(5, model.Files.Count);
        Assert.Equal(EngineType.QwenAsr, model.EngineType);
        Assert.Null(model.SherpaModelType);
        Assert.Contains(model.Files, f => f.LocalName == "model.safetensors");
        Assert.Contains(model.Files, f => f.LocalName == "vocab.json");
    }

    [Fact]
    public void Qwen3AsrOnnx_HasExpectedFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "qwen3-asr-0.6b-onnx");
        Assert.Equal(5, model.Files.Count);
        Assert.Equal(EngineType.QwenAsrOnnx, model.EngineType);
        Assert.Null(model.SherpaModelType);
        Assert.Contains(model.Files, f => f.LocalName == "encoder.onnx");
        Assert.Contains(model.Files, f => f.LocalName == "decoder.onnx");
        Assert.Contains(model.Files, f => f.LocalName == "embed_tokens.bin");
    }

    [Fact]
    public void WindowsSpeech_HasNoFiles()
    {
        var model = ModelInfo.AvailableModels.First(m => m.Id == "windows-speech");
        Assert.Empty(model.Files);
        Assert.Equal(EngineType.WindowsSpeech, model.EngineType);
        Assert.Null(model.SherpaModelType);
    }

    [Fact]
    public void InferenceMethod_ReturnsCorrectStrings()
    {
        var whisper = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.WhisperCpp);
        Assert.Contains("whisper.cpp", whisper.InferenceMethod);

        var sherpaOffline = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.SherpaOnnxOffline);
        Assert.Contains("sherpa-onnx offline", sherpaOffline.InferenceMethod);

        var sherpaStreaming = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.SherpaOnnxStreaming);
        Assert.Contains("sherpa-onnx streaming", sherpaStreaming.InferenceMethod);

        var qwenAsr = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.QwenAsr);
        Assert.Contains("qwen-asr", qwenAsr.InferenceMethod);

        var qwenAsrOnnx = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.QwenAsrOnnx);
        Assert.Contains("ONNX Runtime/DirectML", qwenAsrOnnx.InferenceMethod);

        var windowsSpeech = ModelInfo.AvailableModels.First(m => m.EngineType == EngineType.WindowsSpeech);
        Assert.Contains("Windows Speech API", windowsSpeech.InferenceMethod);
    }
}
