"""Step 3 — train a Gaussian splat of the environment from the tracked frames.

Backends (all free):
  brush      github.com/ArthurBrussee/brush — prebuilt Windows/macOS/Linux
             binaries on its releases page, runs on NVIDIA/AMD/Intel via WebGPU.
             Default.  No CPU mode.
  opensplat  github.com/pierotofy/OpenSplat — source only (no prebuilt
             binaries); has a CPU build if you compile it yourself.
  custom     any command; use {input} {output} {iters} placeholders in
             settings.splat_command, e.g. a nerfstudio splatfacto wrapper.

The input is the undistorted COLMAP project from the track step
(track/undistorted with images/ and sparse/0/).  Output is splat/splat.ply plus
splat_preview.json (subsampled centres + colours for the viewer).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError
from ..plyio import read_ply, splat_to_points

PREVIEW_POINTS = 120_000


@register
class Splat(Step):
    name = "splat"
    label = "Splat"
    depends = ["track"]
    params = ["splat_backend", "splat_iterations", "splat_command", "device"]
    outputs = ["splat.ply", "splat_preview.json"]

    def preflight(self, project):
        b = project.settings.get("splat_backend", "opensplat")
        if b == "custom":
            return [] if project.settings.get("splat_command") else ["splat_command is empty"]
        if b == "brush" and not project.tool("brush"):
            return ["brush not found — download brush-app-x86_64-pc-windows-msvc.zip from github.com/ArthurBrussee/brush/releases and run:  python -m vfxpipe tools --brush <path to the .exe>"]
        if b == "opensplat" and not project.tool("opensplat"):
            return ["opensplat not found — OpenSplat has no prebuilt binaries; build it from source or switch splat_backend to brush"]
        if b == "opensplat" and project.device() == "cpu":
            return []  # allowed, just slow
        return []

    def run(self, ctx: Context):
        s = ctx.settings
        inp = ctx.project.dir("track") / "undistorted"
        _ensure_sparse0(inp)
        out_ply = ctx.out / "splat.ply"
        if out_ply.exists():
            out_ply.unlink()
        iters = int(s.get("splat_iterations", 7000))
        backend = s.get("splat_backend", "opensplat")
        ctx.set_progress(0.01, f"training with {backend} ({iters} iterations)")

        if backend == "custom":
            cmd = [c.format(input=str(inp), output=str(out_ply), iters=iters)
                   for c in _split(s["splat_command"])]
            ctx.run(cmd, cwd=ctx.out, progress_parser=lambda l: _iter_prog(l, iters))
        elif backend == "brush":
            exe = ctx.project.tool("brush")
            cmd = [exe, str(inp), "--total-steps", str(iters), "--export-every", str(iters),
                   "--export-path", str(ctx.out), "--export-name", "splat.ply"]
            ctx.run(cmd, cwd=ctx.out, progress_parser=lambda l: _iter_prog(l, iters))
        else:  # opensplat
            exe = ctx.project.tool("opensplat")
            cmd = [exe, str(inp), "-n", str(iters), "-o", str(out_ply)]
            if ctx.project.device() == "cpu":
                cmd += ["--cpu"]
            ctx.run(cmd, cwd=ctx.out, progress_parser=lambda l: _iter_prog(l, iters))

        if not out_ply.exists():
            # some tools name the export differently; grab the newest .ply
            plys = sorted(ctx.out.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
            if not plys:
                raise StepError("training finished but no .ply was produced — check logs/splat.log")
            shutil.copy(plys[-1], out_ply)
        ctx.set_progress(0.95, "writing preview")
        n = write_preview(out_ply, ctx.out / "splat_preview.json")
        ctx.log(f"splat has {n} gaussians")
        return {"gaussians": n, "backend": backend, "iterations": iters}


def write_preview(ply: Path, dest: Path) -> int:
    verts, _ = read_ply(ply)
    xyz, rgb, op, sc = splat_to_points(verts)
    n = len(xyz)
    keep = op > 0.3
    idx = np.where(keep)[0]
    if len(idx) > PREVIEW_POINTS:
        idx = np.random.default_rng(0).choice(idx, PREVIEW_POINTS, replace=False)
    dest.write_text(json.dumps({
        "xyz": xyz[idx].round(4).tolist(),
        "rgb": (rgb[idx] * 255).astype(int).tolist(),
        "total": int(n),
    }))
    return n


def _ensure_sparse0(undist: Path) -> None:
    """OpenSplat/Brush look for <project>/sparse/0/*.bin — COLMAP's undistorter writes sparse/*.bin."""
    src = undist / "sparse"
    dst = src / "0"
    if not dst.exists() and any(src.glob("*.bin")):
        dst.mkdir()
        for f in src.glob("*.bin"):
            shutil.copy(f, dst / f.name)


def _iter_prog(line: str, total: int):
    m = re.search(r"(?:Step|step|iter|Iteration)\D{0,4}(\d+)", line)
    if m:
        return 0.01 + 0.93 * min(1.0, int(m.group(1)) / max(1, total))
    return None


def _split(cmd: str) -> list[str]:
    import shlex
    return shlex.split(cmd, posix=False)
