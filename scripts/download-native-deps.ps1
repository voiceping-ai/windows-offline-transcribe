<#
.SYNOPSIS
  Download pre-built native DLLs required by the app (whisper.cpp, sherpa-onnx).
  Called by CI and by developers who don't want to compile from source.

.DESCRIPTION
  Downloads and extracts:
  - whisper.cpp (whisper.dll, ggml.dll, ggml-base.dll, ggml-cpu.dll)
  - sherpa-onnx  (sherpa-onnx-c-api.dll, onnxruntime.dll, onnxruntime_providers_shared.dll)

  Output: libs/runtimes/win-x64/*.dll
#>

param(
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\libs\runtimes\win-x64')
)

$ErrorActionPreference = 'Stop'

# ── Versions ──
$WhisperVersion  = '1.8.3'
$SherpaVersion   = '1.12.25'

# ── URLs ──
$WhisperUrl = "https://github.com/ggml-org/whisper.cpp/releases/download/v${WhisperVersion}/whisper-bin-x64.zip"
$SherpaUrl  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SherpaVersion}/sherpa-onnx-v${SherpaVersion}-win-x64-shared.tar.bz2"

# ── Setup ──
$tempDir = Join-Path $env:TEMP 'native-deps-download'
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Output directory: $OutputDir"

# ── Download whisper.cpp ──
Write-Host "`n=== Downloading whisper.cpp v${WhisperVersion} ==="
$whisperZip = Join-Path $tempDir 'whisper-bin-x64.zip'
Invoke-WebRequest -Uri $WhisperUrl -OutFile $whisperZip -UseBasicParsing
Write-Host "  Downloaded: $([math]::Round((Get-Item $whisperZip).Length / 1MB, 1)) MB"

$whisperExtract = Join-Path $tempDir 'whisper'
Expand-Archive -Path $whisperZip -DestinationPath $whisperExtract -Force

# Copy DLLs (they may be in root or in a bin/ subfolder)
$whisperDlls = @('whisper.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll')
foreach ($dll in $whisperDlls) {
    $found = Get-ChildItem -Path $whisperExtract -Filter $dll -Recurse | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName (Join-Path $OutputDir $dll) -Force
        Write-Host "  Copied: $dll ($([math]::Round($found.Length / 1KB)) KB)"
    } else {
        Write-Warning "  Not found: $dll"
    }
}

# ── Download sherpa-onnx ──
Write-Host "`n=== Downloading sherpa-onnx v${SherpaVersion} ==="
$sherpaTar = Join-Path $tempDir 'sherpa-onnx.tar.bz2'

# Try the standard URL first, fall back to -MD-Release variant
try {
    Invoke-WebRequest -Uri $SherpaUrl -OutFile $sherpaTar -UseBasicParsing
} catch {
    Write-Host "  Standard URL failed, trying MD-Release variant..."
    $SherpaUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SherpaVersion}/sherpa-onnx-v${SherpaVersion}-win-x64-shared-MD-Release.tar.bz2"
    Invoke-WebRequest -Uri $SherpaUrl -OutFile $sherpaTar -UseBasicParsing
}
Write-Host "  Downloaded: $([math]::Round((Get-Item $sherpaTar).Length / 1MB, 1)) MB"

# Extract tar.bz2 (use tar which is built into Windows 10+)
$sherpaExtract = Join-Path $tempDir 'sherpa'
New-Item -ItemType Directory -Force -Path $sherpaExtract | Out-Null
tar -xf $sherpaTar -C $sherpaExtract 2>&1 | Out-Null

$sherpaDlls = @('sherpa-onnx-c-api.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll')
foreach ($dll in $sherpaDlls) {
    $found = Get-ChildItem -Path $sherpaExtract -Filter $dll -Recurse | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName (Join-Path $OutputDir $dll) -Force
        Write-Host "  Copied: $dll ($([math]::Round($found.Length / 1KB)) KB)"
    } else {
        Write-Warning "  Not found: $dll"
    }
}

# ── Cleanup ──
Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue

# ── Verify ──
Write-Host "`n=== Verification ==="
$required = @('whisper.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll',
              'sherpa-onnx-c-api.dll', 'onnxruntime.dll')
$missing = @()
foreach ($dll in $required) {
    $path = Join-Path $OutputDir $dll
    if (Test-Path $path) {
        Write-Host "  OK: $dll ($([math]::Round((Get-Item $path).Length / 1KB)) KB)"
    } else {
        Write-Host "  MISSING: $dll"
        $missing += $dll
    }
}

if ($missing.Count -gt 0) {
    throw "Missing required DLLs: $($missing -join ', ')"
}

Write-Host "`nAll native dependencies downloaded successfully."
