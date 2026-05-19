$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Source = Join-Path $ProjectRoot "dist\PubReader"
$OutputDir = Join-Path $ProjectRoot "output"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\PubReader"
$DesktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Pub Reader.lnk"
$ExePath = Join-Path $InstallDir "PubReader.exe"
$ConfigDir = Join-Path $env:USERPROFILE ".pub_reader"
$ConfigPath = Join-Path $ConfigDir "config.json"

if (-not (Test-Path $Source)) {
    throw "Build folder not found: $Source. Run scripts\build_windows.ps1 first."
}

if (Test-Path $InstallDir) {
    Remove-Item -Recurse -Force -LiteralPath $InstallDir
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Recurse -Force -Path (Join-Path $Source "*") -Destination $InstallDir
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

$Config = [ordered]@{
    api_base_url = "https://api.deepseek.com/chat/completions"
    model_name = "deepseek-v4-pro"
    library_dir = $OutputDir
}
$Config | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -Path $ConfigPath

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($DesktopShortcut)
$Shortcut.TargetPath = $ExePath
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "Pub Reader"
$Shortcut.Save()

Write-Host "Pub Reader installed to $InstallDir"
Write-Host "Output folder configured: $OutputDir"
Write-Host "Desktop shortcut created: $DesktopShortcut"
