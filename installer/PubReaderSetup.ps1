$ErrorActionPreference = "Stop"

$Source = Join-Path $PSScriptRoot "..\dist\PubReader"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\PubReader"
$DesktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Pub Reader.lnk"
$ExePath = Join-Path $InstallDir "PubReader.exe"

if (-not (Test-Path $Source)) {
    throw "Build folder not found: $Source. Run scripts\build_windows.ps1 first."
}

if (Test-Path $InstallDir) {
    Remove-Item -Recurse -Force -LiteralPath $InstallDir
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Recurse -Force -Path (Join-Path $Source "*") -Destination $InstallDir

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($DesktopShortcut)
$Shortcut.TargetPath = $ExePath
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "Pub Reader"
$Shortcut.Save()

Write-Host "Pub Reader installed to $InstallDir"
Write-Host "Desktop shortcut created: $DesktopShortcut"
