"""A project is one shot: one folder holding the source footage reference,
settings, and one sub-folder of outputs per pipeline step.

    shot01/
      project.json      settings (this file is the single source of truth)
      frames/           000001.jpg ...
      track/            COLMAP database, sparse model, cameras.json, points.json
      splat/            splat.ply, splat_preview.json
      mesh/             mesh.ply, mesh_preview.json
      lighting/         cubemap faces, env.hdr, lookdev.png
      roto/             objects.json, masks/<object>/000001.png
      export/           scene.blend, build log
      cache/            per-step input hashes + status
      logs/             per-step logs
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

STEP_DIRS = ["frames", "track", "splat", "mesh", "lighting", "roto", "export", "cache", "logs"]

DEFAULT_SETTINGS: dict[str, Any] = {
    "source_video": "",          # absolute path to the footage
    "in_frame": 0,               # trim (in source frames, 0-based)
    "out_frame": -1,             # -1 = until the end
    "max_width": 1920,           # downscale frames for the pipeline (0 = keep)
    "frame_format": "jpg",       # jpg is what SAM 2 expects; png also fine
    "jpg_quality": 3,            # ffmpeg -q:v (2 best .. 31 worst)
    "device": "auto",            # auto | cuda | cpu
    # --- track ---
    "camera_model": "OPENCV",    # SIMPLE_RADIAL | RADIAL | OPENCV | OPENCV_FISHEYE
    "sequential_overlap": 15,    # frames matched to each other in a video
    "loop_detection": False,        # needs a vocab tree file
    "use_glomap": True,          # fall back to COLMAP's mapper if glomap missing
    "auto_orient": True,         # rotate so 'up' becomes +Z
    "orient_mode": "floor",      # floor (largest horizontal plane in the points) | camera (average camera up)
    "up_vector": [0, 0, 0],      # manual override in COLMAP coords (0,0,0 = off); set from a plane in the Mesh panel
    "world_scale": 1.0,          # multiply the solve so 1 unit = 1 metre (measure a door)
    # --- splat ---
    "splat_backend": "brush",      # brush (prebuilt Windows binary, any GPU) | opensplat (build from source) | custom
    "splat_iterations": 7000,
    "splat_command": "",         # custom command template, see steps/splat.py
    # --- mesh ---
    "mesh_mode": "planes",       # planes (RANSAC floor/walls/tables) | poisson (over splat points) | colmap_dense (multi-view stereo)
    "dense_max_size": 1000,      # colmap_dense: image size for stereo (speed vs detail)
    "dense_trim": 7,             # colmap_dense: Poisson trim (higher = fewer invented surfaces)
    "plane_max": 8,
    "plane_min_inliers": 3.0,    # % of remaining points a plane must claim
    "plane_tolerance": 1.0,      # % of scene size
    "mesh_source": "splat",      # splat | sparse
    "mesh_min_opacity": 0.6,     # keep gaussians at least this opaque (surfaces, not haze)
    "mesh_max_scale_pct": 90,    # drop the largest 10% of gaussians (blobs between surfaces)
    "poisson_depth": 9,
    "mesh_density_trim": 0.05,   # drop lowest-density 5% of Poisson output (unseen areas)
    "mesh_target_faces": 200000,
    # --- lighting ---
    "env_position": [0.0, 0.0, 0.0],  # where the character stands (COLMAP world coords)
    "env_face_size": 512,
    "env_highlight_boost": 4.0,  # multiply pixels near clip to fake HDR range
    "env_highlight_threshold": 0.9,
    # --- roto ---
    "sam2_model": "auto",        # auto | tiny | small | base_plus | large
    # --- export ---
    "render_engine": "CYCLES",
    "tools": {                   # empty = look on PATH
        "ffmpeg": "", "ffprobe": "", "colmap": "", "glomap": "",
        "blender": "", "opensplat": "", "brush": "",
    },
}

TOOL_NAMES = list(DEFAULT_SETTINGS["tools"].keys())
APP_ROOT = Path(__file__).resolve().parent.parent          # the folder with run.bat
LOCAL_TOOLS = APP_ROOT / "tools.json"                          # self-contained: travels with the folder
USER_TOOLS = Path.home() / ".vfxpipe" / "tools.json"           # optional per-user fallback
MACHINE_TOOLS = LOCAL_TOOLS
def _unwrap_bat(path: str) -> str:
    """COLMAP's Windows release ships COLMAP.bat, which breaks when the path
    contains '&' (cmd.exe treats it as a command separator).  Use bin/colmap.exe
    directly; runtime DLL folders are added to PATH by Context.run."""
    p = Path(path)
    if p.suffix.lower() in (".bat", ".cmd"):
        stem = p.stem.lower()
        for cand in (p.parent / "bin" / f"{stem}.exe", p.parent / f"{stem}.exe"):
            if cand.is_file():
                return str(cand)
    return path


def tool_env(exe: str) -> dict[str, str]:
    """Environment for launching exe: its own folder and the sibling lib/ folder
    (COLMAP layout) go on PATH so DLLs resolve without the .bat wrapper."""
    env = dict(os.environ)
    d = Path(exe).resolve().parent
    extra = [d, d.parent / "lib", d / "lib"]
    env["PATH"] = os.pathsep.join(str(x) for x in extra if x.is_dir()) + os.pathsep + env.get("PATH", "")
    return env


TOOL_ALIASES = {"brush": ["brush_app", "brush"], "colmap": ["colmap", "COLMAP"]}


def _read_tools(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return {k: v for k, v in json.loads(path.read_text()).items() if k in TOOL_NAMES and v}
        except json.JSONDecodeError:
            pass
    return {}


def load_machine_tools() -> dict[str, str]:
    """tools.json next to run.bat wins over the per-user one."""
    return {**_read_tools(USER_TOOLS), **_read_tools(LOCAL_TOOLS)}


def save_machine_tools(tools: dict[str, str], user: bool = False) -> Path:
    path = USER_TOOLS if user else LOCAL_TOOLS
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_tools(path)
    current.update({k: v for k, v in tools.items() if k in TOOL_NAMES})
    path.write_text(json.dumps(current, indent=2))
    return path


class Project:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for d in STEP_DIRS:
            (self.root / d).mkdir(exist_ok=True)
        self.settings_path = self.root / "project.json"
        self.settings: dict[str, Any] = json.loads(json.dumps(DEFAULT_SETTINGS))
        if self.settings_path.exists():
            self._merge(json.loads(self.settings_path.read_text()))
        else:
            self.save()

    # ---- settings -------------------------------------------------------
    def _merge(self, incoming: dict[str, Any]) -> None:
        for k, v in incoming.items():
            if k == "tools" and isinstance(v, dict):
                self.settings["tools"].update(v)
            else:
                self.settings[k] = v

    def update(self, incoming: dict[str, Any]) -> None:
        self._merge(incoming)
        self.save()

    def save(self) -> None:
        self.settings_path.write_text(json.dumps(self.settings, indent=2))

    def dir(self, name: str) -> Path:
        return self.root / name

    # ---- tools ----------------------------------------------------------
    def tool(self, name: str) -> str | None:
        """Resolved executable for a tool, or None if it can't be found."""
        found = self._tool(name)
        return _unwrap_bat(found) if found else None

    def _tool(self, name: str) -> str | None:
        configured = (self.settings.get("tools") or {}).get(name, "") or load_machine_tools().get(name, "")
        if configured:
            p = Path(configured)
            if p.is_dir():  # someone pointed at the folder — look inside
                for alias in TOOL_ALIASES.get(name, [name]):
                    for cand in (p / f"{alias}.exe", p / alias, p / f"{alias}.bat", p / "bin" / f"{alias}.exe"):
                        if cand.is_file():
                            return str(cand)
            elif p.is_file():
                return str(p)
            found = shutil.which(configured)
            if found:
                return found
        for alias in TOOL_ALIASES.get(name, [name]):
            found = shutil.which(alias)
            if found:
                return found
        if name == "ffprobe":  # usually next to ffmpeg
            ff = self.tool("ffmpeg")
            if ff:
                for cand in (Path(ff).with_name("ffprobe.exe"), Path(ff).with_name("ffprobe")):
                    if cand.is_file():
                        return str(cand)
        # Common Windows install spots
        if os.name == "nt":
            candidates = {
                "blender": [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"],
                "colmap": [Path("C:/colmap"), Path("C:/COLMAP")],
                "glomap": [Path("C:/glomap"), Path("C:/colmap")],
            }.get(name, [])
            for base in candidates:
                if base.exists():
                    hits = sorted(base.rglob(f"{name}.exe"))
                    if hits:
                        return str(hits[-1])
        return None

    def tool_status(self) -> dict[str, str | None]:
        return {n: self.tool(n) for n in TOOL_NAMES}

    # ---- device ---------------------------------------------------------
    def device(self) -> str:
        want = self.settings.get("device", "auto")
        if want == "cpu":
            return "cpu"
        try:
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu" if want == "auto" else want

    def has_nvidia(self) -> bool:
        """An NVIDIA driver is present (nvidia-smi) — enough for COLMAP/Brush even without torch."""
        if self.settings.get("device") == "cpu":
            return False
        if shutil.which("nvidia-smi"):
            return True
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def gpu_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"torch": False, "cuda": False, "name": None}
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda"] = torch.cuda.is_available()
            if info["cuda"]:
                info["name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        if not info["name"] and shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                info["name"] = out.splitlines()[0] if out else None
            except Exception:
                pass
        return info

    # ---- frames ---------------------------------------------------------
    def frame_files(self) -> list[Path]:
        ext = self.settings.get("frame_format", "jpg")
        return sorted(self.dir("frames").glob(f"*.{ext}"))

    def frame_count(self) -> int:
        return len(self.frame_files())

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "settings": self.settings,
            "tools": self.tool_status(),
            "device": self.device(),
            "gpu": self.gpu_info(),
            "frame_count": self.frame_count(),
        }
