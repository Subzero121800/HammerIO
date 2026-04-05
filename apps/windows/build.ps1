# HammerIO Desktop Application — Windows Build
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Builds a Windows installer using PyInstaller or creates a portable bundle.
# GPU compression requires NVIDIA CUDA GPU + drivers.
#
# Prerequisites:
#   - Python 3.10+ from python.org
#   - pip install pyinstaller
#
# Usage:
#   .\apps\windows\build.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$Version = "1.0.0"
$BuildDir = Join-Path $ScriptDir "build"
$AppName = "HammerIO"

Write-Host ""
Write-Host "  ================================================"
Write-Host "    HammerIO Desktop App Builder - Windows"
Write-Host "    Version: $Version"
Write-Host "  ================================================"
Write-Host ""

# Clean
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir | Out-Null

# ─── Create portable bundle ──────────────────────────────────────────────────

$BundleDir = Join-Path $BuildDir "$AppName-$Version-windows"
New-Item -ItemType Directory -Path $BundleDir | Out-Null

# Copy source
$Exclude = @('.git', '__pycache__', '.venv', '*.pyc', '.pytest_cache', '*.egg-info', 'Test Video.mp4')
Copy-Item -Path "$ProjectDir\*" -Destination $BundleDir -Recurse -Exclude $Exclude

# Create launcher batch files
@"
@echo off
title HammerIO
echo Starting HammerIO...

if not exist "%~dp0.venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv"
    "%~dp0.venv\Scripts\pip" install -e ".[all]"
)

"%~dp0.venv\Scripts\python" -m hammerio %*
"@ | Out-File -FilePath (Join-Path $BundleDir "hammer.bat") -Encoding ASCII

@"
@echo off
title HammerIO Dashboard
echo Starting HammerIO Dashboard...
echo Open http://localhost:5000 in your browser

if not exist "%~dp0.venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv"
    "%~dp0.venv\Scripts\pip" install -e ".[web]"
)

start http://localhost:5000
"%~dp0.venv\Scripts\python" "%~dp0start_webui.py"
"@ | Out-File -FilePath (Join-Path $BundleDir "HammerIO Dashboard.bat") -Encoding ASCII

@"
@echo off
echo Installing HammerIO...
python -m venv "%~dp0.venv"
"%~dp0.venv\Scripts\pip" install --upgrade pip wheel
"%~dp0.venv\Scripts\pip" install -e ".[all]"
echo.
echo HammerIO installed! Run hammer.bat or HammerIO Dashboard.bat
pause
"@ | Out-File -FilePath (Join-Path $BundleDir "install.bat") -Encoding ASCII

# Create zip
$ZipPath = Join-Path $BuildDir "$AppName-$Version-windows.zip"
Compress-Archive -Path $BundleDir -DestinationPath $ZipPath

Write-Host ""
Write-Host "  Windows bundle built:"
Write-Host "    $ZipPath"
Write-Host ""
Write-Host "  To install: Extract zip, run install.bat"
Write-Host "  To use:     hammer.bat compress file.csv"
Write-Host "  Dashboard:  HammerIO Dashboard.bat"
Write-Host ""
Write-Host "  Note: GPU acceleration requires NVIDIA GPU + CUDA drivers."
