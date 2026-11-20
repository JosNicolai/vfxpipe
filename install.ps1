# vfxpipe - Windows setup.  Run from the vfxpipe folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\install.ps1
# Optional:  .\install.ps1 -Cpu            (torch without CUDA - roto runs on CPU)
#            .\install.ps1 -Cuda cu128     (force a CUDA build: cu118, cu126 or cu128)
# By default the CUDA build is chosen from your GPU (nvidia-smi).
param([switch]$Cpu, [string]$Cuda = "")

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

if ((Get-Location).Path -match "OneDrive") {
  Write-Host "WARNING: this folder is inside OneDrive. The .venv has thousands of small files and OneDrive can turn them"
  Write-Host "         into online-only placeholders, which breaks Python and the tools. Moving the project to e.g. C:\vfxpipe is safer."
}

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

# 2. torch - pick the CUDA build that matches the GPU
if (-not $Cpu -and -not $Cuda) {
  $cap = $null
  try { $cap = (& nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1) } catch {}
  if ($cap) {
    Write-Host "GPU: $cap"
    $ver = [double](($cap -split ",")[1].Trim())
    if ($ver -ge 12.0) { $Cuda = "cu128" }        # RTX 50-series (Blackwell) needs CUDA 12.8 builds
    else { $Cuda = "cu126" }                       # GTX 10 / RTX 20 / 30 / 40
  } else {
    Write-Host "No NVIDIA GPU / nvidia-smi found - installing CPU torch (roto will be slower)."
    $Cpu = $true
  }
}
if ($Cpu) {
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
  Write-Host "Installing torch for $Cuda"
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
}
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# 3. SAM 2 (roto)
$env:SAM2_BUILD_CUDA = "0"   # optional CUDA extension is not needed
pip install "git+https://github.com/facebookresearch/sam2.git"

Write-Host ""
Write-Host "Python side done. Next: install the external tools (ffmpeg, COLMAP, GLOMAP, Brush) automatically with:"
Write-Host "    .\setup-tools.ps1"
Write-Host "Blender 4.2 LTS or newer is not installed by this; an existing install is found automatically."
Write-Host ""
Write-Host "Start:   .\run.bat"
