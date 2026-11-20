"""Step 2 — camera track.

    colmap feature_extractor  →  colmap sequential_matcher  →  glomap mapper
    →  colmap image_undistorter  →  model_converter (TXT)  →  cameras.json / points.json

Everything after this step works on the *undistorted* plates in
track/undistorted/images, which are pinhole and line up with a normal Blender
camera.  The original frames stay in frames/ untouched.

GPU: SIFT extraction and matching use CUDA when available.  On CPU they are
roughly 5-10x slower but give identical results.  GLOMAP itself is CPU-bound.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError
from .. import colmap_io

MAX_POINTS_PREVIEW = 150_000
MAX_OBS_PER_FRAME = 600


@register
class Track(Step):
    name = "track"
    label = "Camera track"
    depends = ["frames"]
    params = ["camera_model", "sequential_overlap", "loop_detection", "vocab_tree_path",
              "use_glomap", "auto_orient", "orient_mode", "up_vector", "world_scale", "device"]
    outputs = ["cameras.json", "points.json", "undistorted/images"]

    def preflight(self, project):
        probs = []
        if not project.tool("colmap"):
            probs.append("colmap not found — download the Windows release from github.com/colmap/colmap/releases and set tools.colmap")
        if project.settings.get("use_glomap") and not project.tool("glomap"):
            probs.append("glomap not found (set tools.glomap, or turn off use_glomap to use COLMAP's slower mapper)")
        if project.settings.get("loop_detection") and not project.settings.get("vocab_tree_path"):
            probs.append("loop_detection needs vocab_tree_path (download vocab_tree_flickr100K_words32K.bin from demuc.de/colmap)")
        return probs

    def run(self, ctx: Context):
        s = ctx.settings
        colmap = ctx.project.tool("colmap")
        glomap = ctx.project.tool("glomap")
        frames = ctx.project.dir("frames")
        out = ctx.out
        db = out / "database.db"
        sparse = out / "sparse"
        undist = out / "undistorted"
        for p in (db, sparse, undist):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        sparse.mkdir()
        gpu = "1" if ctx.project.has_nvidia() else "0"
        ctx.log("using GPU for SIFT:", gpu == "1", "(set device=cpu to force CPU)")

        n_frames = ctx.project.frame_count()
        ctx.set_progress(0.02, "extracting features")
        ctx.run([colmap, "feature_extractor",
                 "--database_path", db, "--image_path", frames,
                 "--ImageReader.camera_model", s["camera_model"],
                 "--ImageReader.single_camera", "1",
                 "--SiftExtraction.use_gpu", gpu,
                 "--SiftExtraction.max_num_features", "8192"],
                progress_parser=lambda l: _feat_prog(l, n_frames))
        ctx.check_cancel()

        ctx.set_progress(0.25, "matching (sequential)")
        cmd = [colmap, "sequential_matcher", "--database_path", db,
               "--SequentialMatching.overlap", str(int(s.get("sequential_overlap", 15))),
               "--SiftMatching.use_gpu", gpu]
        if s.get("loop_detection") and s.get("vocab_tree_path"):
            cmd += ["--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", s["vocab_tree_path"]]
        ctx.run(cmd)
        ctx.check_cancel()

        ctx.set_progress(0.5, "solving cameras")
        if s.get("use_glomap") and glomap:
            ctx.run([glomap, "mapper", "--database_path", db, "--image_path", frames,
                     "--output_path", sparse])
        else:
            ctx.run([colmap, "mapper", "--database_path", db, "--image_path", frames,
                     "--output_path", sparse])
        model_dir = _pick_model(sparse)
        if model_dir is None:
            raise StepError("no reconstruction produced — try more overlap, a different camera_model, or check the footage has parallax")
        ctx.log("using model", model_dir)

        ctx.set_progress(0.75, "undistorting plates")
        ctx.run([colmap, "image_undistorter", "--image_path", frames, "--input_path", model_dir,
                 "--output_path", undist, "--output_type", "COLMAP"])
        ctx.run([colmap, "model_converter", "--input_path", undist / "sparse",
                 "--output_path", undist / "sparse", "--output_type", "TXT"])
        ctx.check_cancel()

        ctx.set_progress(0.9, "writing cameras.json")
        model = colmap_io.read_model(undist / "sparse")
        up = s.get("up_vector") or [0, 0, 0]
        xform = colmap_io.world_transform(model, bool(s.get("auto_orient", True)), float(s.get("world_scale", 1.0)),
                                          mode=s.get("orient_mode", "floor"), manual_up=up if any(up) else None)
        ctx.log(f"orientation from {xform['up_source']} ({xform['floor_points']} floor points); "
                f"camera was tilted {xform['camera_tilt_deg']}° from that up")
        o = ctx.project.dir("mesh") / "orientation.json"
        if o.exists():
            o.unlink()  # a fresh solve invalidates the refined orientation
        summary = self._export(ctx, model, xform, n_frames)
        return summary

    # ---- export -----------------------------------------------------------
    def _export(self, ctx: Context, model: colmap_io.Model, xform: dict, n_frames: int) -> dict:
        idx = model.id_index()
        by_name = {img.name: img for img in model.images.values()}
        frames_meta = []
        solved = 0
        errs = []
        for k in range(1, n_frames + 1):
            ext = ctx.settings.get("frame_format", "jpg")
            name = f"{k:06d}.{ext}"
            img = by_name.get(name)
            if img is None:
                frames_meta.append({"frame": k, "name": name, "solved": False})
                continue
            cam = model.cameras[img.camera_id]
            fx, fy, cx, cy = cam.K()
            valid = img.point3D_ids >= 0
            obs_ids = img.point3D_ids[valid]
            obs_xy = img.xys[valid]
            # reprojection error of this frame's observations
            R, t = img.R(), img.tvec
            keep = np.array([pid in idx for pid in obs_ids], dtype=bool) if len(obs_ids) else np.zeros(0, bool)
            err = None
            if keep.any():
                P = model.xyz[[idx[int(p)] for p in obs_ids[keep]]]
                Xc = (R @ P.T).T + t
                z = Xc[:, 2]
                ok = z > 1e-6
                if ok.any():
                    u = fx * Xc[ok, 0] / z[ok] + cx
                    v = fy * Xc[ok, 1] / z[ok] + cy
                    d = np.hypot(u - obs_xy[keep][ok, 0], v - obs_xy[keep][ok, 1])
                    err = float(np.mean(d))
                    errs.append(err)
            # subsample observations for the viewer
            n = int(keep.sum())
            sel = np.where(keep)[0]
            if n > MAX_OBS_PER_FRAME:
                sel = np.random.default_rng(k).choice(sel, MAX_OBS_PER_FRAME, replace=False)
            obs = [[round(float(obs_xy[i, 0]), 1), round(float(obs_xy[i, 1]), 1), int(obs_ids[i])] for i in sel]
            solved += 1
            frames_meta.append({
                "frame": k, "name": name, "solved": True,
                "w": cam.width, "h": cam.height,
                "K": [fx, fy, cx, cy],
                "R": R.round(8).tolist(), "t": t.round(6).tolist(),
                "C": img.center().round(6).tolist(),
                "n_obs": int(valid.sum()), "err": None if err is None else round(err, 3),
                "obs": obs,
            })
        cameras = {"transform": xform, "frames": frames_meta,
                   "n_frames": n_frames, "n_solved": solved,
                   "mean_err": float(np.mean(errs)) if errs else None}
        (ctx.out / "cameras.json").write_text(json.dumps(cameras))

        # points: keep everything the observations reference, plus a random fill
        N = len(model.xyz)
        take = np.arange(N)
        if N > MAX_POINTS_PREVIEW:
            take = np.random.default_rng(0).choice(N, MAX_POINTS_PREVIEW, replace=False)
        pts = {"ids": model.ids[take].tolist(),
               "xyz": model.xyz[take].round(5).tolist(),
               "rgb": model.rgb[take].tolist(),
               "err": model.error[take].round(3).tolist(),
               "total": N}
        (ctx.out / "points.json").write_text(json.dumps(pts))
        _write_ply(ctx.out / "points.ply", model.xyz, model.rgb)
        ctx.log(f"solved {solved}/{n_frames} frames, {N} points, mean reprojection error "
                f"{cameras['mean_err']:.3f}px" if errs else f"solved {solved}/{n_frames}")
        return {"solved": solved, "frames": n_frames, "points": N, "mean_err": cameras["mean_err"],
                "estimated_up": xform["estimated_up"]}


def _feat_prog(line: str, n: int):
    m = re.search(r"Processed file \[(\d+)/(\d+)\]", line)
    if m:
        return 0.02 + 0.23 * int(m.group(1)) / max(1, int(m.group(2)))
    return None


def _pick_model(sparse: Path) -> Path | None:
    """GLOMAP/COLMAP write sparse/0, sparse/1 ... — pick the one with most images."""
    best, best_n = None, -1
    for d in sorted(sparse.iterdir()) if sparse.exists() else []:
        if not d.is_dir():
            continue
        imgs = d / "images.bin"
        n = imgs.stat().st_size if imgs.exists() else -1
        if n > best_n:
            best, best_n = d, n
    return best


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())
