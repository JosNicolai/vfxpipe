# vfxpipe - Windows setup.  Run from the vfxpipe folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\install.ps1
# Optional:  .\install.ps1 -Cpu     (torch without CUDA - roto runs on CPU)
param([switch]$Cpu)

$ErrorActionPreference = "Stop"

# --- find a real Python (not the Microsoft Store one, whose venv hangs) ----------
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($v in "3.12", "3.11", "3.10") {
    $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $exe -and $exe -notmatch "WindowsApps") { $py = @("py", "-$v"); break }
  }
}
if (-not $py) {
  $exe = (Get-Command python -ErrorAction SilentlyContinue).Source
  if ($exe -and $exe -notmatch "WindowsApps") { $py = @("python") }
}
if (-not $py) {
  Write-Host ""
  Write-Host "No usable Python found. The Microsoft Store Python cannot create virtual environments reliably."
  Write-Host "Install Python 3.12 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH'),"
  Write-Host "then run this script again. The 'py' launcher it installs is picked up automatically."
  exit 1
}
Write-Host "Using Python: $(& $py[0] $py[1..($py.Length)] -c 'import sys; print(sys.executable)')"

# --- venv -------------------------------------------------------------------------
if (Test-Path .venv) {
  if (-not (Test-Path .venv\Scripts\python.exe)) { Remove-Item -Recurse -Force .venv }   # half-made venv from a failed run
}
if (-not (Test-Path .venv)) { & $py[0] $py[1..($py.Length)] -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

# 1. core + mesh
pip install -e .
pip install open3d

# 2. torch - CUDA 12.x build for the 3060 Ti, or CPU
if ($Cpu) {
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

# 3. SAM 2 (roto)
$env:SAM2_BUILD_CUDA = "0"   # optional CUDA extension is not needed
pip install "git+https://github.com/facebookresearch/sam2.git"

Write-Host ""
Write-Host "Python side done. External tools (all free) - use existing installs or unzip anywhere, then run:  python -m vfxpipe tools --colmap ... --brush ..."
Write-Host "  ffmpeg      https://www.gyan.dev/ffmpeg/builds/          (add bin\ to PATH)"
Write-Host "  COLMAP      https://github.com/colmap/colmap/releases    (colmap.bat / colmap.exe, CUDA build)"
Write-Host "  GLOMAP      https://github.com/colmap/glomap/releases    (glomap.exe)"
Write-Host "  Brush       https://github.com/ArthurBrussee/brush/releases     (brush-app-x86_64-pc-windows-msvc.zip)"
Write-Host "  Blender     https://www.blender.org/download/            (4.2 LTS or newer; an existing install is found automatically)"
Write-Host ""
Write-Host "Start:   run.bat D:\shots\shot01 D:\footage\take03.mp4"
