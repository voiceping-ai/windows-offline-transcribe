<#
.SYNOPSIS
  Build native translation DLLs (CTranslate2, SentencePiece, NativeTranslation wrapper) from source.

.DESCRIPTION
  Clones and builds:
  1. SentencePiece v0.2.1 (static library)
  2. CTranslate2 v4.7.1 (shared library / DLL) with OpenBLAS backend
  3. OfflineTranscription.NativeTranslation.dll (our C ABI wrapper)

  Downloads OpenBLAS development headers/libs from OpenMathLib releases.

  Output: libs/runtimes/win-x64/
    - OfflineTranscription.NativeTranslation.dll
    - ctranslate2.dll

  Prerequisites:
    - C++ compiler on PATH: either MSVC (cl.exe) or MinGW (g++.exe)
    - CMake 3.21+ on PATH
    - Git on PATH
    - Ninja on PATH (used with MinGW; MSVC uses Visual Studio generator)

.PARAMETER Config
  Build configuration: Release (default) or Debug.

.PARAMETER BuildDir
  Temporary build directory. Defaults to <repo>\build\translation-deps.

.PARAMETER OutputDir
  Where to copy final DLLs. Defaults to <repo>\libs\runtimes\win-x64.

.PARAMETER Clean
  Remove build directory before starting.

.EXAMPLE
  .\scripts\build-translation-deps.ps1 -Config Release
  .\scripts\build-translation-deps.ps1 -Config Release -Clean
#>

param(
    [ValidateSet('Release', 'Debug')]
    [string]$Config = 'Release',

    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build\translation-deps'),

    [string]$OutputDir = (Join-Path $PSScriptRoot '..\libs\runtimes\win-x64'),

    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# -- Versions --
$SentencePieceVersion = 'v0.2.1'
$CTranslate2Version   = 'v4.7.1'
$OpenBLASVersion      = '0.3.28'

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

# -- Resolve paths --
$BuildDir  = [System.IO.Path]::GetFullPath($BuildDir)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

Write-Host "============================================"
Write-Host "  Translation Native Dependencies Builder"
Write-Host "============================================"
Write-Host "Config:    $Config"
Write-Host "BuildDir:  $BuildDir"
Write-Host "OutputDir: $OutputDir"
Write-Host ""

# -- Prerequisite checks --
function Assert-Command($cmd, $label) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "$label ($cmd) is required but not found on PATH."
    }
}
Assert-Command 'cmake' 'CMake 3.21+'
Assert-Command 'git'   'Git'

# -- Detect compiler toolchain --
# Prefer MSVC if available (Visual Studio generator), fall back to MinGW + Ninja.
$useMSVC = $false
$cmakeGenerator = $null
$cmakeExtraArgs = @()

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if ((Test-Path $vsWhere)) {
    $vsInstall = & $vsWhere -products '*' -latest -property installationPath -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 2>$null
    if ($vsInstall) {
        $useMSVC = $true
        $cmakeGenerator = 'Visual Studio 17 2022'
        $cmakeExtraArgs = @('-A', 'x64')
        Write-Host "Compiler: MSVC (Visual Studio at $vsInstall)"
    }
}

if (-not $useMSVC) {
    # Try MinGW g++
    if (Get-Command 'g++' -ErrorAction SilentlyContinue) {
        # Prefer Ninja if available, otherwise MinGW Makefiles
        if (Get-Command 'ninja' -ErrorAction SilentlyContinue) {
            $cmakeGenerator = 'Ninja'
            Write-Host "Compiler: MinGW GCC + Ninja"
        } else {
            $cmakeGenerator = 'MinGW Makefiles'
            Write-Host "Compiler: MinGW GCC + MinGW Makefiles"
        }
    } else {
        throw "No C++ compiler found. Install Visual Studio with C++ workload or MinGW (g++.exe on PATH)."
    }
}

# For single-config generators (Ninja, Makefiles), CMAKE_BUILD_TYPE is set at configure time.
# For multi-config generators (Visual Studio), --config is used at build time.
$isSingleConfig = ($cmakeGenerator -eq 'Ninja' -or $cmakeGenerator -eq 'MinGW Makefiles')

# -- Clean if requested --
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "`nCleaning build directory..."
    Remove-Item -Recurse -Force $BuildDir
}

# -- Create directories --
New-Item -ItemType Directory -Force -Path $BuildDir  | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir  | Out-Null

# ================================================================
# Step 1: Download OpenBLAS development package (headers + .lib)
# ================================================================
Write-Host "`n=== Step 1: OpenBLAS development package ==="

$openblasDir    = Join-Path $BuildDir 'openblas'
$openblasZip    = Join-Path $BuildDir 'openblas-dev.zip'
$openblasUrl    = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OpenBLASVersion}/OpenBLAS-${OpenBLASVersion}-x64.zip"

if (-not (Test-Path (Join-Path $openblasDir 'lib'))) {
    Write-Host "  Downloading OpenBLAS v${OpenBLASVersion} dev package..."
    Invoke-WebRequest -Uri $openblasUrl -OutFile $openblasZip -UseBasicParsing
    Write-Host "  Downloaded: $([math]::Round((Get-Item $openblasZip).Length / 1MB, 1)) MB"

    New-Item -ItemType Directory -Force -Path $openblasDir | Out-Null
    Expand-Archive -Path $openblasZip -DestinationPath $openblasDir -Force
    Write-Host "  Extracted OpenBLAS dev package."
} else {
    Write-Host "  OpenBLAS dev package already present, skipping download."
}

# Find the actual include/lib dirs (may be nested)
$openblasInclude = Get-ChildItem -Path $openblasDir -Filter 'cblas.h' -Recurse | Select-Object -First 1
if (-not $openblasInclude) { throw "OpenBLAS headers not found after extraction." }
$openblasIncludeDir = $openblasInclude.DirectoryName

$openblasLibFile = Get-ChildItem -Path $openblasDir -Filter 'libopenblas.lib' -Recurse | Select-Object -First 1
if (-not $openblasLibFile) {
    $openblasLibFile = Get-ChildItem -Path $openblasDir -Filter 'openblas.lib' -Recurse | Select-Object -First 1
}
if (-not $openblasLibFile) {
    # MinGW: look for libopenblas.a or libopenblas.dll.a
    $openblasLibFile = Get-ChildItem -Path $openblasDir -Filter 'libopenblas.dll.a' -Recurse | Select-Object -First 1
    if (-not $openblasLibFile) {
        $openblasLibFile = Get-ChildItem -Path $openblasDir -Filter 'libopenblas.a' -Recurse | Select-Object -First 1
    }
}
if (-not $openblasLibFile) { throw "OpenBLAS library file not found after extraction." }
$openblasLibDir = $openblasLibFile.DirectoryName

Write-Host "  OpenBLAS include: $openblasIncludeDir"
Write-Host "  OpenBLAS lib:     $openblasLibDir ($($openblasLibFile.Name))"

# ================================================================
# Step 2: Build SentencePiece (static library)
# ================================================================
Write-Host "`n=== Step 2: SentencePiece $SentencePieceVersion (static) ==="

$spSrcDir   = Join-Path $BuildDir 'sentencepiece-src'
$spBuildDir = Join-Path $BuildDir 'sentencepiece-build'
$spInstDir  = Join-Path $BuildDir 'sentencepiece-install'

if (-not (Test-Path $spSrcDir)) {
    Write-Host "  Cloning SentencePiece..."
    git clone --depth 1 --branch $SentencePieceVersion `
        'https://github.com/google/sentencepiece.git' $spSrcDir
} else {
    Write-Host "  SentencePiece source already cloned."
}

if (-not (Test-Path (Join-Path $spInstDir 'lib'))) {
    Write-Host "  Configuring SentencePiece..."
    $spCmakeArgs = @(
        '-S', $spSrcDir,
        '-B', $spBuildDir,
        '-G', $cmakeGenerator
    ) + $cmakeExtraArgs + @(
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DCMAKE_INSTALL_PREFIX=$spInstDir",
        '-DSPM_ENABLE_SHARED=OFF',
        '-DSPM_BUILD_TEST=OFF',
        '-DSPM_USE_BUILTIN_PROTOBUF=ON'
    )
    & cmake @spCmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "SentencePiece configure failed (exit $LASTEXITCODE)" }

    Write-Host "  Building SentencePiece..."
    $spBuildArgs = @('--build', $spBuildDir, '--parallel')
    if (-not $isSingleConfig) { $spBuildArgs += @('--config', $Config) }
    & cmake @spBuildArgs
    if ($LASTEXITCODE -ne 0) { throw "SentencePiece build failed (exit $LASTEXITCODE)" }

    Write-Host "  Installing SentencePiece..."
    $spInstArgs = @('--install', $spBuildDir)
    if (-not $isSingleConfig) { $spInstArgs += @('--config', $Config) }
    & cmake @spInstArgs
    if ($LASTEXITCODE -ne 0) { throw "SentencePiece install failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "  SentencePiece already built."
}

# Generate cmake config for SentencePiece (it doesn't install one by default)
$spCmakeConfigDir = Join-Path $spInstDir 'lib\cmake\sentencepiece'
if (-not (Test-Path (Join-Path $spCmakeConfigDir 'sentencepieceConfig.cmake'))) {
    Write-Host "  Generating SentencePiece cmake config..."
    New-Item -ItemType Directory -Force -Path $spCmakeConfigDir | Out-Null
    $spCmakeConfig = @"
# Generated cmake config for SentencePiece (static library)
if(TARGET sentencepiece::sentencepiece)
  return()
endif()

add_library(sentencepiece::sentencepiece STATIC IMPORTED)
set_target_properties(sentencepiece::sentencepiece PROPERTIES
  IMPORTED_LOCATION "$($spInstDir -replace '\\','/')/lib/libsentencepiece.a"
  INTERFACE_INCLUDE_DIRECTORIES "$($spInstDir -replace '\\','/')/include"
)

add_library(sentencepiece::sentencepiece_train STATIC IMPORTED)
set_target_properties(sentencepiece::sentencepiece_train PROPERTIES
  IMPORTED_LOCATION "$($spInstDir -replace '\\','/')/lib/libsentencepiece_train.a"
  INTERFACE_INCLUDE_DIRECTORIES "$($spInstDir -replace '\\','/')/include"
)
"@
    Set-Content -Path (Join-Path $spCmakeConfigDir 'sentencepieceConfig.cmake') -Value $spCmakeConfig -Encoding ascii
}

# ================================================================
# Step 3: Build CTranslate2 (shared library / DLL)
# ================================================================
Write-Host "`n=== Step 3: CTranslate2 $CTranslate2Version (shared, OpenBLAS backend) ==="

$ct2SrcDir   = Join-Path $BuildDir 'ctranslate2-src'
$ct2BuildDir = Join-Path $BuildDir 'ctranslate2-build'
$ct2InstDir  = Join-Path $BuildDir 'ctranslate2-install'

if (-not (Test-Path $ct2SrcDir)) {
    Write-Host "  Cloning CTranslate2 (with submodules)..."
    git clone --depth 1 --recurse-submodules --shallow-submodules --branch $CTranslate2Version `
        'https://github.com/OpenNMT/CTranslate2.git' $ct2SrcDir
} else {
    Write-Host "  CTranslate2 source already cloned."
}

if (-not (Test-Path (Join-Path $ct2InstDir 'lib'))) {
    Write-Host "  Configuring CTranslate2..."
    $ct2CmakeArgs = @(
        '-S', $ct2SrcDir,
        '-B', $ct2BuildDir,
        '-G', $cmakeGenerator
    ) + $cmakeExtraArgs + @(
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DCMAKE_INSTALL_PREFIX=$ct2InstDir",
        '-DBUILD_SHARED_LIBS=ON',
        '-DWITH_MKL=OFF',
        '-DWITH_OPENBLAS=ON',
        '-DWITH_CUDA=OFF',
        '-DWITH_CUDNN=OFF',
        '-DWITH_DNNL=OFF',
        '-DOPENMP_RUNTIME=COMP',
        '-DENABLE_CPU_DISPATCH=OFF',
        '-DBUILD_CLI=OFF',
        '-DBUILD_TESTS=OFF',
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
        "-DOPENBLAS_INCLUDE_DIR=$openblasIncludeDir",
        "-DOPENBLAS_LIBRARY=$($openblasLibFile.FullName)",
        "-DCMAKE_PREFIX_PATH=$spInstDir"
    )
    & cmake @ct2CmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CTranslate2 configure failed (exit $LASTEXITCODE)" }

    Write-Host "  Building CTranslate2 (this may take several minutes)..."
    $ct2BuildArgs = @('--build', $ct2BuildDir, '--parallel')
    if (-not $isSingleConfig) { $ct2BuildArgs += @('--config', $Config) }
    & cmake @ct2BuildArgs
    if ($LASTEXITCODE -ne 0) { throw "CTranslate2 build failed (exit $LASTEXITCODE)" }

    Write-Host "  Installing CTranslate2..."
    $ct2InstArgs = @('--install', $ct2BuildDir)
    if (-not $isSingleConfig) { $ct2InstArgs += @('--config', $Config) }
    & cmake @ct2InstArgs
    if ($LASTEXITCODE -ne 0) { throw "CTranslate2 install failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "  CTranslate2 already built."
}

# ================================================================
# Step 4: Build NativeTranslation wrapper DLL
# ================================================================
Write-Host "`n=== Step 4: OfflineTranscription.NativeTranslation wrapper ==="

$ntSrcDir   = Join-Path $RepoRoot 'src\OfflineTranscription.NativeTranslation'
$ntBuildDir = Join-Path $BuildDir 'native-translation-build'

Write-Host "  Configuring NativeTranslation wrapper..."
$ntCmakeArgs = @(
    '-S', $ntSrcDir,
    '-B', $ntBuildDir,
    '-G', $cmakeGenerator
) + $cmakeExtraArgs + @(
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_PREFIX_PATH=$ct2InstDir;$spInstDir"
)
& cmake @ntCmakeArgs
if ($LASTEXITCODE -ne 0) { throw "NativeTranslation configure failed (exit $LASTEXITCODE)" }

Write-Host "  Building NativeTranslation wrapper..."
$ntBuildArgs = @('--build', $ntBuildDir, '--parallel')
if (-not $isSingleConfig) { $ntBuildArgs += @('--config', $Config) }
& cmake @ntBuildArgs
if ($LASTEXITCODE -ne 0) { throw "NativeTranslation build failed (exit $LASTEXITCODE)" }

# ================================================================
# Step 5: Copy output DLLs
# ================================================================
Write-Host "`n=== Step 5: Copying DLLs to $OutputDir ==="

# Find and copy our wrapper DLL
$wrapperDll = Get-ChildItem -Path $ntBuildDir -Filter 'OfflineTranscription.NativeTranslation.dll' -Recurse | Select-Object -First 1
# Also check for libOfflineTranscription.NativeTranslation.dll (MinGW prefixes with lib)
if (-not $wrapperDll) {
    $wrapperDll = Get-ChildItem -Path $ntBuildDir -Filter 'libOfflineTranscription.NativeTranslation.dll' -Recurse | Select-Object -First 1
}
if (-not $wrapperDll) { throw "OfflineTranscription.NativeTranslation.dll not found in build output." }
Copy-Item $wrapperDll.FullName (Join-Path $OutputDir 'OfflineTranscription.NativeTranslation.dll') -Force
Write-Host "  Copied: OfflineTranscription.NativeTranslation.dll ($([math]::Round($wrapperDll.Length / 1KB)) KB)"

# Find and copy ctranslate2.dll
$ct2Dll = Get-ChildItem -Path $ct2InstDir -Filter 'ctranslate2.dll' -Recurse | Select-Object -First 1
if (-not $ct2Dll) {
    $ct2Dll = Get-ChildItem -Path $ct2InstDir -Filter 'libctranslate2.dll' -Recurse | Select-Object -First 1
}
if (-not $ct2Dll) {
    $ct2Dll = Get-ChildItem -Path $ct2BuildDir -Filter 'ctranslate2.dll' -Recurse | Select-Object -First 1
}
if (-not $ct2Dll) {
    $ct2Dll = Get-ChildItem -Path $ct2BuildDir -Filter 'libctranslate2.dll' -Recurse | Select-Object -First 1
}
if (-not $ct2Dll) { throw "ctranslate2.dll not found in install or build output." }
Copy-Item $ct2Dll.FullName (Join-Path $OutputDir 'libctranslate2.dll') -Force
Write-Host "  Copied: libctranslate2.dll ($([math]::Round($ct2Dll.Length / 1KB)) KB)"

# Copy MinGW runtime DLLs that ctranslate2.dll depends on (if not already present)
if (-not $useMSVC) {
    $gccPath = (Get-Command 'gcc' -ErrorAction SilentlyContinue).Source
    if ($gccPath) {
        $mingwBinDir = Split-Path $gccPath -Parent
        $mingwRuntimeDlls = @('libstdc++-6.dll', 'libgomp-1.dll')
        foreach ($dll in $mingwRuntimeDlls) {
            $dest = Join-Path $OutputDir $dll
            if (-not (Test-Path $dest)) {
                $src = Join-Path $mingwBinDir $dll
                if (Test-Path $src) {
                    Copy-Item $src $dest -Force
                    Write-Host "  Copied: $dll (MinGW runtime, $([math]::Round((Get-Item $src).Length / 1KB)) KB)"
                } else {
                    Write-Warning "  MinGW runtime $dll not found at $src"
                }
            } else {
                Write-Host "  OK: $dll (already present)"
            }
        }
    }
}

# ================================================================
# Verification
# ================================================================
Write-Host "`n=== Verification ==="

$requiredDlls = @(
    'OfflineTranscription.NativeTranslation.dll',
    'libctranslate2.dll'
)
$missing = @()
foreach ($dll in $requiredDlls) {
    $path = Join-Path $OutputDir $dll
    if (Test-Path $path) {
        Write-Host "  OK: $dll ($([math]::Round((Get-Item $path).Length / 1KB)) KB)"
    } else {
        Write-Host "  MISSING: $dll"
        $missing += $dll
    }
}

if ($missing.Count -gt 0) {
    throw "Build failed - missing DLLs: $($missing -join ', ')"
}

# Verify libopenblas.dll is already present (shared with qwen_asr)
$openblasRuntime = Join-Path $OutputDir 'libopenblas.dll'
if (Test-Path $openblasRuntime) {
    Write-Host "  OK: libopenblas.dll (already present, shared with qwen_asr)"
} else {
    Write-Warning "  libopenblas.dll not found in output dir - ctranslate2.dll requires it at runtime."
    Write-Warning "  Copy libopenblas.dll to $OutputDir or ensure it is on PATH."
}

Write-Host "`nTranslation native dependencies built successfully."
Write-Host "Output: $OutputDir"
