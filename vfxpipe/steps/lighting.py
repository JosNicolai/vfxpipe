"""Step 5 — an environment map from the splat, seen from where the character
will stand.

This is a deliberately simple CPU renderer: every gaussian is drawn as a small
depth-sorted disc into six cube faces, holes are filled with a push-pull
pyramid, the cube is resampled to an equirectangular panorama, near-clipped
pixels are boosted to fake some HDR range, and the result is saved as
lighting/env.hdr (Radiance RGBE, which Blender reads directly).

It is not a true HDRI — the footage's clipped windows and lamps stay clipped —
but it is the correct colour, direction and rough intensity of everything in
the room, which is most of what matters for a fur character.  Shoot a real
bracketed HDRI when you can and drop it in as lighting/env.hdr; the export step
uses whatever is there.

If Blender is found, a lookdev render (grey ball + chrome ball) is produced so
you can judge the lighting before any character is involved.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError
from ..plyio import read_ply, splat_to_points


@register
class Lighting(Step):
    name = "lighting"
    label = "Lighting"
    depends = ["splat"]
    params = ["env_position", "env_face_size", "env_highlight_boost", "env_highlight_threshold"]

    def extra_hash(self, project):
        from ..runner import load_status
        return "mesh:" + load_status(project, "mesh").get("output_hash", "")
    outputs = ["env.hdr", "env_preview.png"]

    def run(self, ctx: Context):
        s = ctx.settings
        from ..colmap_io import load_transform
        cams = json.loads((ctx.project.dir("track") / "cameras.json").read_text())
        M = np.array(load_transform(ctx.project)["M"])
        verts, _ = read_ply(ctx.project.dir("splat") / "splat.ply")
        xyz, rgb, op, sc = splat_to_points(verts)
        keep = op > 0.2
        xyz, rgb, sc = xyz[keep], rgb[keep], sc[keep]

        rgb = srgb_to_linear(rgb)   # splat colours are display values; lighting needs linear light
        pos = np.array(s.get("env_position") or [0, 0, 0], dtype=np.float64)
        if not s.get("env_position") or np.allclose(pos, 0):
            pos = default_env_position(cams, ctx.project.dir("track") / "points.json")
            ctx.log("env_position not set — using the point the camera looks at most:", pos.round(3).tolist())
        # into Z-up world so the panorama needs no rotation in Blender
        P = (M @ (xyz - pos).T).T
        S = int(s.get("env_face_size", 512))
        ctx.set_progress(0.05, "splatting cube faces")
        faces = render_cubemap(P, rgb, sc.max(axis=1), S, ctx)
        ctx.set_progress(0.7, "building panorama")
        eq = cubemap_to_equirect(faces, S)
        boost, thr = float(s.get("env_highlight_boost", 4.0)), float(s.get("env_highlight_threshold", 0.9))
        lum = eq @ np.array([0.2126, 0.7152, 0.0722])
        thr = float(srgb_to_linear(np.array([thr]))[0])          # threshold is given in display terms
        ramp = np.clip((lum - thr) / max(1e-6, 1 - thr), 0, 1)
        eq = eq * (1 + boost * ramp)[..., None]
        write_hdr(ctx.out / "env.hdr", eq.astype(np.float32))
        from PIL import Image
        prev = np.clip(eq, 0, 1) ** (1 / 2.2)
        Image.fromarray((prev * 255).astype(np.uint8)).save(ctx.out / "env_preview.png")
        (ctx.out / "faces").mkdir(exist_ok=True)
        for name, img in faces.items():
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(ctx.out / "faces" / f"{name}.png")
        (ctx.out / "lighting.json").write_text(json.dumps({"env_position": pos.tolist(), "face_size": S}))

        blender = ctx.project.tool("blender")
        if blender:
            ctx.set_progress(0.85, "rendering lookdev in Blender")
            script = Path(__file__).resolve().parent.parent / "blender" / "render_lookdev.py"
            try:
                ctx.run([blender, "-b", "--python", str(script), "--",
                         str(ctx.out / "env.hdr"), str(ctx.out / "lookdev.png"),
                         "GPU" if ctx.project.device() == "cuda" else "CPU"])
            except StepError as e:
                ctx.log("lookdev render failed (non-fatal):", e)
        else:
            ctx.log("blender not found — skipping lookdev render")
        return {"env_position": pos.tolist(), "face_size": S, "lookdev": (ctx.out / "lookdev.png").exists()}


# ---- helpers ------------------------------------------------------------------

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def default_env_position(cams: dict, points_json: Path | None = None) -> np.ndarray:
    """Median over frames of 'camera centre + forward * median depth of what
    that frame sees' — i.e. roughly the spot the camera was pointed at."""
    lookup = {}
    if points_json and points_json.exists():
        pts = json.loads(points_json.read_text())
        lookup = {int(i): np.array(p) for i, p in zip(pts["ids"], pts["xyz"])}
    out = []
    for f in cams["frames"]:
        if not f.get("solved"):
            continue
        R, C = np.array(f["R"]), np.array(f["C"])
        fwd = R.T @ np.array([0, 0, 1.0])
        depths = [float((R @ lookup[o[2]] + np.array(f["t"]))[2]) for o in f.get("obs", []) if o[2] in lookup]
        d = 0.7 * (float(np.median(depths)) if depths else 2.0)  # a bit in front of the median surface
        out.append(C + fwd * d)
    if not out:
        return np.zeros(3)
    return np.median(np.array(out), axis=0)


# cube faces: name -> (forward, right, up)  in Z-up world
FACES = {
    "px": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "nx": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "py": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "ny": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "pz": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "nz": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
}


def render_cubemap(P: np.ndarray, rgb: np.ndarray, radius_world: np.ndarray, S: int, ctx=None) -> dict[str, np.ndarray]:
    out = {}
    depth_all = np.linalg.norm(P, axis=1) + 1e-9
    for i, (name, (fwd, right, up)) in enumerate(FACES.items()):
        if ctx:
            ctx.set_progress(0.05 + 0.6 * i / 6, f"face {name}")
        fwd, right, up = map(lambda v: np.array(v, dtype=np.float64), (fwd, right, up))
        z = P @ fwd
        ok = z > 1e-6
        x = (P[ok] @ right) / z[ok]
        y = (P[ok] @ up) / z[ok]
        inside = (np.abs(x) <= 1.15) & (np.abs(y) <= 1.15)
        x, y = x[inside], y[inside]
        d = depth_all[ok][inside]
        col = rgb[ok][inside]
        u = (x * 0.5 + 0.5) * S
        v = (0.5 - y * 0.5) * S
        rpx = np.clip(radius_world[ok][inside] / d * S * 0.5, 0.5, 4.0)
        # expand each point into a small disc: bins of radius 0..4 px
        us, vs, ds, cs = [], [], [], []
        for r in (0, 1, 2, 3, 4):
            sel = (np.round(rpx).astype(int) == r) if r < 4 else (np.round(rpx) >= 4)
            if not sel.any():
                continue
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    if du * du + dv * dv > r * r + r:
                        continue
                    us.append(u[sel] + du); vs.append(v[sel] + dv); ds.append(d[sel]); cs.append(col[sel])
        if not us:
            out[name] = np.zeros((S, S, 3))
            continue
        u = np.concatenate(us); v = np.concatenate(vs); d = np.concatenate(ds); col = np.concatenate(cs)
        ui = np.floor(u).astype(int); vi = np.floor(v).astype(int)
        m = (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
        ui, vi, d, col = ui[m], vi[m], d[m], col[m]
        flat = vi * S + ui
        order = np.argsort(-d)             # far → near
        flat_o = flat[order][::-1]         # near first
        _, first = np.unique(flat_o, return_index=True)
        win = order[::-1][first]           # index of nearest point per pixel
        img = np.zeros((S * S, 3))
        cov = np.zeros(S * S, dtype=bool)
        img[flat[win]] = col[win]
        cov[flat[win]] = True
        img = push_pull(img.reshape(S, S, 3), cov.reshape(S, S))
        out[name] = img
    return out


def push_pull(img: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fill uncovered pixels from coarser levels (Gortler et al. push-pull)."""
    levels = [(img * cov[..., None], cov.astype(np.float64))]
    while levels[-1][0].shape[0] > 4:
        c, w = levels[-1]
        h, wd = c.shape[0] // 2, c.shape[1] // 2
        c2 = c[:2 * h:2, :2 * wd:2] + c[1:2 * h:2, :2 * wd:2] + c[:2 * h:2, 1:2 * wd:2] + c[1:2 * h:2, 1:2 * wd:2]
        w2 = w[:2 * h:2, :2 * wd:2] + w[1:2 * h:2, :2 * wd:2] + w[:2 * h:2, 1:2 * wd:2] + w[1:2 * h:2, 1:2 * wd:2]
        levels.append((c2, w2))
    # pull: normalise coarsest, then fill finer levels
    c, w = levels[-1]
    filled = c / np.maximum(w, 1e-9)[..., None]
    # a cell nothing reached at all gets the average of everything that was seen
    if (w <= 0).any():
        mean = c.sum(axis=(0, 1)) / max(1e-9, w.sum())
        filled[w <= 0] = mean
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)[:c.shape[0], :c.shape[1]]
        wn = np.clip(w, 0, 1)[..., None]
        filled = (c / np.maximum(w, 1e-9)[..., None]) * wn + up * (1 - wn)
    return filled


def cubemap_to_equirect(faces: dict[str, np.ndarray], S: int) -> np.ndarray:
    W, H = 4 * S, 2 * S
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    # Blender's convention: +X at the image centre, +Y a quarter in from the left,
    # +Z at the top (manual: camera rotation (90,0,-90) looks at the centre).
    phi = (0.5 - U) * 2 * np.pi
    theta = (V - 0.5) * np.pi
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    D = np.stack([x, y, z], axis=-1)
    out = np.zeros((H, W, 3))
    ax = np.argmax(np.abs(D), axis=-1)
    sign = np.sign(np.take_along_axis(D, ax[..., None], axis=-1)[..., 0])
    for name, (fwd, right, up) in FACES.items():
        fwd, right, up = map(lambda t: np.array(t, dtype=np.float64), (fwd, right, up))
        a = int(np.argmax(np.abs(fwd)))
        sel = (ax == a) & (sign == np.sign(fwd[a]))
        d = D[sel]
        zc = d @ fwd
        xc = (d @ right) / zc
        yc = (d @ up) / zc
        ui = np.clip(((xc * 0.5 + 0.5) * S).astype(int), 0, S - 1)
        vi = np.clip(((0.5 - yc * 0.5) * S).astype(int), 0, S - 1)
        out[sel] = faces[name][vi, ui]
    return out


def write_hdr(path: Path, img: np.ndarray) -> None:
    """Radiance RGBE, uncompressed scanlines."""
    H, W, _ = img.shape
    img = np.maximum(img, 0).astype(np.float32)
    m = img.max(axis=-1)
    e = np.zeros_like(m, dtype=np.int32)
    nz = m > 1e-32
    mant, exp = np.frexp(m[nz])
    scale = mant * 256.0 / m[nz]
    rgbe = np.zeros((H, W, 4), dtype=np.uint8)
    rgbe[..., :3][nz] = np.clip(img[nz] * scale[..., None], 0, 255).astype(np.uint8)
    e[nz] = exp + 128
    rgbe[..., 3] = e.astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {H} +X {W}\n".encode())
        f.write(rgbe.tobytes())
