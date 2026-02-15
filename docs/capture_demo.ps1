Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinApi {
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, IntPtr dwExtraInfo);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    public const uint MOUSEEVENTF_LEFTUP = 0x0004;
}
"@

function Capture-Window($hwnd, $path) {
    $rect = New-Object WinApi+RECT
    [WinApi]::GetWindowRect($hwnd, [ref]$rect)
    $w = $rect.Right - $rect.Left
    $h = $rect.Bottom - $rect.Top
    $bitmap = New-Object System.Drawing.Bitmap($w, $h)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, (New-Object System.Drawing.Size($w, $h)))
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
}

function Click-At($hwnd, $relX, $relY) {
    $rect = New-Object WinApi+RECT
    [WinApi]::GetWindowRect($hwnd, [ref]$rect)
    [WinApi]::SetCursorPos($rect.Left + $relX, $rect.Top + $relY)
    Start-Sleep -Milliseconds 100
    [WinApi]::mouse_event([WinApi]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [IntPtr]::Zero)
    [WinApi]::mouse_event([WinApi]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [IntPtr]::Zero)
}

$procs = Get-Process -Name "OfflineTranscription" -ErrorAction SilentlyContinue
if (-not $procs) { Write-Host "App not running"; exit 1 }
$hwnd = $procs[0].MainWindowHandle
[WinApi]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 500

$outDir = "C:\Users\voice\windows-offline-transcribe\docs\images\frames"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# 1. Click transcribe nav to go to main page
Click-At $hwnd 30 95
Start-Sleep -Seconds 1

# Frame 1: Idle
Capture-Window $hwnd "$outDir\frame01.png"
Write-Host "Frame 1: Idle"

# 2. Click file button
Click-At $hwnd 93 669
Start-Sleep -Seconds 2

# Frame 2: File dialog (full screen)
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bmp.Save("$outDir\frame02.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Host "Frame 2: File dialog"

# Type path and enter
[System.Windows.Forms.SendKeys]::SendWait("C:\Users\voice\windows-offline-transcribe\test-english-30s.wav")
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait("{ENTER}")

# Wait for transcription
Start-Sleep -Seconds 8

# Frame 3: Result
[WinApi]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 500
Capture-Window $hwnd "$outDir\frame03.png"
Write-Host "Frame 3: Transcription result"
Write-Host "Done"
