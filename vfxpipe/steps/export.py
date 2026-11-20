"""Step 7 — export.

Always writes export/bundle/ — plain files any 3D app can import:
    camera.json      per-frame Blender-convention camera (location, quaternion, lens, shift)
    planes.obj       fitted planes, one object each, Z-up and scaled   (if the Mesh step ran in planes mode)
    proxy.ply        Poisson / dense mesh, Z-up and scaled              (other mesh modes)
    points.ply       sparse track points, Z-up and scaled
    env.hdr          environment map (Blender orientation)
    masks/           roto mattes (copied)
    import_bundle.py run inside Blender (Scripting tab → Open → Run Script) to set up
                     the animated camera, plate background, planes, points and world
If Blender is found, export/scene.blend is built as well; a failure there is
logged but doesn't fail the step.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError, load_status
from ..plyio import read_ply, write_mesh_ply


@register
class Export(Step):
    name = "export"
    label = "Export"
    depends = ["track"]
    params = ["render_engine", "world_scale", "auto_orient", "frame_format"]
    outputs = ["bundle", "scene.blend"]

    def extra_hash(self, project):
        return "|".join(load_status(project, s).get("output_hash", "") for s in ("mesh", "lighting", "roto"))

    def run(self, ctx: Context):
        root = ctx.project.root
        bundle = ctx.out / "bundle"
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir()
        from ..colmap_io import load_transform
        cams = json.loads((root / "track" / "cameras.json").read_text())
        T = load_transform(ctx.project); M = np.array(T["M"]); c = np.array(T["centre"]); sc = float(T["scale"])
        to_world = lambda X: (M @ (np.asarray(X, dtype=float) - c)) * sc  # noqa: E731

        # --- camera -------------------------------------------------------------
        flip = np.diag([1.0, -1.0, -1.0])
        frames = []
        for f in cams["frames"]:
            if not f.get("solved"):
                continue
            R = np.array(f["R"]); w, h = f["w"], f["h"]; fx, fy, cx, cy = f["K"]
            rot = M @ R.T @ flip
            frames.append({"frame": f["frame"], "location": to_world(f["C"]).round(6).tolist(),
                           "rotation_matrix": rot.round(8).tolist(), "quaternion": _mat2quat(rot).round(8).tolist(),
                           "lens_mm": round(fx / w * 36.0, 5), "sensor_width_mm": 36.0,
                           "shift_x": round((w / 2 - cx) / max(w, h), 6), "shift_y": round((cy - h / 2) / max(w, h), 6)})
        fj = root / "frames" / "frames.json"
        fm = json.loads(fj.read_text()) if fj.exists() else {}
        plate_dir = root / "track" / "undistorted" / "images"
        (bundle / "camera.json").write_text(json.dumps({
            "convention": "Blender: Z up, camera looks down -Z, +Y up in camera space; units = solve units * world_scale",
            "width": cams["frames"][frames[0]["frame"] - 1]["w"], "height": cams["frames"][frames[0]["frame"] - 1]["h"],
            "fps": fm.get("fps", 25), "frame_start": 1, "frame_end": cams["n_frames"],
            "plate_dir": str(plate_dir), "plate_pattern": f"%06d.{ctx.settings.get('frame_format', 'jpg')}",
            "character_position": (to_world(json.loads((root / "lighting" / "lighting.json").read_text())["env_position"]).round(6).tolist()
                                   if (root / "lighting" / "lighting.json").exists() else None),
            "frames": frames}, indent=1))
        ctx.log(f"camera.json: {len(frames)} frames")

        # --- geometry -------------------------------------------------------------
        planes = root / "mesh" / "planes.json"
        if planes.exists():
            lines = ["# vfxpipe fitted planes (Z up, scaled)"]; vi = 1
            for pl in json.loads(planes.read_text())["planes"]:
                lines.append(f"o {pl['name']}")
                for cnr in pl["corners"]:
                    x, y, z = to_world(cnr); lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
                lines.append(f"f {vi} {vi + 1} {vi + 2} {vi + 3}"); vi += 4
            (bundle / "planes.obj").write_text("\n".join(lines) + "\n")
            ctx.log("planes.obj written")
        elif (root / "mesh" / "mesh.ply").exists():
            V, F = read_ply(root / "mesh" / "mesh.ply")
            xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(float)
            write_mesh_ply(bundle / "proxy.ply", ((M @ (xyz - c).T).T * sc), F if F is not None else np.zeros((0, 3), int))
            ctx.log("proxy.ply written")
        pts = root / "track" / "points.ply"
        if pts.exists():
            V, _ = read_ply(pts)
            xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(float)
            rgb = np.stack([V["red"], V["green"], V["blue"]], 1)
            write_mesh_ply(bundle / "points.ply", ((M @ (xyz - c).T).T * sc), np.zeros((0, 3), int), rgb)
        env = root / "lighting" / "env.hdr"
        if env.exists():
            shutil.copy(env, bundle / "env.hdr")
        masks = root / "roto" / "masks"
        if masks.exists():
            shutil.copytree(masks, bundle / "masks")
        shutil.copy(Path(__file__).resolve().parent.parent / "blender" / "import_bundle.py", bundle / "import_bundle.py")

        # --- optional .blend -----------------------------------------------------------
        blend_ok = False
        blender = ctx.project.tool("blender")
        if blender:
            ctx.set_progress(0.6, "building scene.blend")
            script = Path(__file__).resolve().parent.parent / "blender" / "build_scene.py"
            try:
                ctx.run([blender, "-b", "--python", str(script), "--", str(root)])
                blend_ok = (ctx.out / "scene.blend").exists()
            except StepError as e:
                ctx.log("scene.blend failed (bundle is still complete):", e)
        else:
            ctx.log("blender not found — bundle only")
        return {"bundle": str(bundle), "files": sorted(p.name for p in bundle.iterdir()), "blend": blend_ok}


def _mat2quat(R: np.ndarray) -> np.ndarray:
    m = R; t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])
