# Windows Offline Transcribe

WinUI 3 (Windows App SDK) desktop app for offline speech transcription.

## Prereqs

- Windows 10/11 (x64)
- Visual Studio 2022 or newer
- .NET 8 SDK

## Native Dependencies

This app uses native engines via P/Invoke:

- whisper.cpp: `whisper.dll` (see `src/OfflineTranscription/Interop/WhisperNative.cs`)
- sherpa-onnx C API: `sherpa-onnx-c-api.dll` (see `src/OfflineTranscription/Interop/SherpaOnnxNative.cs`)

Place required `.dll` files in:

- `libs/runtimes/win-x64/`

They are copied to the build output directory by `src/OfflineTranscription/OfflineTranscription.csproj`.

If you use the sherpa-onnx release bundle, make sure you also include its dependent DLLs
(for example `onnxruntime.dll` and DirectML-related DLLs if applicable).

## Build

Open `OfflineTranscription.sln` and build `x64` (`Debug` or `Release`).

## Notes

- Capture source can be switched in Settings (Microphone vs System Audio / WASAPI loopback).
- For bug reports, Settings includes `Save Screenshot` which writes PNGs under:
  `%LOCALAPPDATA%\\OfflineTranscription\\Diagnostics\\`
- For real-device bug reports, Settings also includes `Evidence Mode`:
  - Logs actions/state to `events.jsonl`
  - Captures screenshots + model file manifests
  - Exports a single ZIP you can share
- See `DEVICE_TESTING.md` for a real-device test checklist.

## Tests

The test project is `tests/OfflineTranscription.Tests/OfflineTranscription.Tests.csproj`.
It compiles non-UI code directly (WinUI projects cannot be referenced from a plain .NET test project).
# windows-offline-transcribe
