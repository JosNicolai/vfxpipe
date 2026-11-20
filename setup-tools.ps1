# vfxpipe - download the external tools (ffmpeg, COLMAP, GLOMAP, Brush) and register them.
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup-tools.ps1                 (everything, CUDA builds when an NVIDIA GPU is found)
#   .\setup-tools.ps1 -NoCuda         (CPU builds of COLMAP / GLOMAP)
#   .\setup-tools.ps1 -Only ffmpeg,brush
#   .\setup-tools.ps1 -Dest D:\vfx-tools
# Tools go to .\tools next to this script, or to %LOCALAPPDATA%\vfxpipe\tools when the project is
# inside OneDrive (OneDrive turns .exe files into online-only placeholders that Windows cannot start).
# Blender is not downloaded: install it yourself (winget install BlenderFoundation.Blender) or use your existing one.
param([string]$Dest = "", [switch]$NoCuda, [string[]]$Only = @("ffmpeg", "colmap", "glomap", "brush"))

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = $PSScriptRoot
if (-not $Dest) {
  if ($root -match "OneDrive") { $Dest = Join-Path $env:LOCALAPPDATA "vfxpipe\tools" } else { $Dest = Join-Path $root "tools" }
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "Installing tools to: $Dest"

$gpu = $false
if (-not $NoCuda) { try { if (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null) { $gpu = $true } } catch {} }
Write-Host ("NVIDIA GPU: " + $gpu)

function Get-ReleaseAsset($repo, [string[]]$include, [string[]]$exclude = @()) {
  $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" -Headers @{ "User-Agent" = "vfxpipe" }
  foreach ($a in $rel.assets) {
    $n = $a.name.ToLower()
    if (-not ($n.EndsWith(".zip"))) { continue }
    $ok = $true
    foreach ($i in $include) { if ($n -notmatch $i) { $ok = $false } }
    foreach ($x in $exclude) { if ($n -match $x) { $ok = $false } }
    if ($ok) { return $a }
  }
  return $null
}

function Install-Zip($name, $url) {
  $dir = Join-Path $Dest $name
  if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
  $zip = Join-Path $Dest ($name + ".zip")
  Write-Host "  downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
  Expand-Archive -Path $zip -DestinationPath $dir -Force
  Remove-Item $zip
  return $dir
}

function Find-Exe($dir, $exe) {
  $f = Get-ChildItem -Path $dir -Recurse -Filter $exe -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($f) { return $f.FullName } else { return $null }
}

$found = @{}
function Do-Tool($key, $label, [scriptblock]$body, $manual) {
  if ($Only -notcontains $key) { return }
  Write-Host "[$label]"
  try { & $body } catch { Write-Host "  FAILED: $($_.Exception.Message)"; Write-Host "  Install it by hand instead: $manual" }
}

Do-Tool "ffmpeg" "ffmpeg" {
  $dir = Install-Zip "ffmpeg" "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
  $exe = Find-Exe $dir "ffmpeg.exe"
  if (-not $exe) { throw "ffmpeg.exe not found in the download" }
  $found["ffmpeg"] = $exe
  Write-Host "  ok: $exe"
} "https://www.gyan.dev/ffmpeg/builds/"

Do-Tool "colmap" "COLMAP" {
  if ($gpu) { $a = Get-ReleaseAsset "colmap/colmap" @("windows", "-cuda") @("nocuda") }
  if (-not $a) { $a = Get-ReleaseAsset "colmap/colmap" @("windows", "nocuda") }
  if (-not $a) { throw "no Windows build found in the latest release" }
  Write-Host "  asset: $($a.name)"
  $dir = Install-Zip "colmap" $a.browser_download_url
  $exe = Find-Exe $dir "colmap.exe"
  if (-not $exe) { throw "colmap.exe not found in the download" }
  $found["colmap"] = $exe
  Write-Host "  ok: $exe"
} "https://github.com/colmap/colmap/releases"

Do-Tool "glomap" "GLOMAP" {
  if ($gpu) { $a = Get-ReleaseAsset "colmap/glomap" @("windows") @("nocuda") }
  if (-not $a) { $a = Get-ReleaseAsset "colmap/glomap" @("windows") }
  if (-not $a) { throw "no Windows build found in the latest release" }
  Write-Host "  asset: $($a.name)"
  $dir = Install-Zip "glomap" $a.browser_download_url
  $exe = Find-Exe $dir "glomap.exe"
  if (-not $exe) { throw "glomap.exe not found in the download" }
  $found["glomap"] = $exe
  Write-Host "  ok: $exe"
} "https://github.com/colmap/glomap/releases"

Do-Tool "brush" "Brush (Gaussian splat trainer, any GPU)" {
  $a = Get-ReleaseAsset "ArthurBrussee/brush" @("windows", "msvc")
  if (-not $a) { $a = Get-ReleaseAsset "ArthurBrussee/brush" @("windows") }
  if (-not $a) { throw "no Windows build found in the latest release" }
  Write-Host "  asset: $($a.name)"
  $dir = Install-Zip "brush" $a.browser_download_url
  $exe = Find-Exe $dir "brush_app.exe"
  if (-not $exe) { $exe = Find-Exe $dir "brush*.exe" }
  if (-not $exe) { throw "brush exe not found in the download" }
  $found["brush"] = $exe
  Write-Host "  ok: $exe"
} "https://github.com/ArthurBrussee/brush/releases"

# register the paths in tools.json (per machine, shared by all shots)
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if ($found.Count -gt 0 -and (Test-Path $venvPy)) {
  $args2 = @("-m", "vfxpipe", "tools")
  foreach ($k in $found.Keys) { $args2 += "--$k"; $args2 += $found[$k] }
  Push-Location $root
  & $venvPy @args2
  Pop-Location
} elseif ($found.Count -gt 0) {
  Write-Host "No .venv yet - run .\install.ps1 first, then this script again to register the paths."
}
Write-Host ""
Write-Host "Check what is found:   .\.venv\Scripts\python.exe -m vfxpipe tools"
Write-Host "Blender 4.2 LTS or newer is found automatically if installed; otherwise:  winget install BlenderFoundation.Blender"
