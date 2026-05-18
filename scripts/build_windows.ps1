$ErrorActionPreference = "Stop"

python -m pip install -e ".[build]"
pyinstaller `
  --noconfirm `
  --windowed `
  --name "PubReader" `
  --add-data "README.md;." `
  "pub_reader\__main__.py"

Copy-Item -Force "installer\PubReaderSetup.ps1" "dist\PubReaderSetup.ps1"
Write-Host "Build finished: dist\PubReader\PubReader.exe"
Write-Host "Setup script copied: dist\PubReaderSetup.ps1"
