"""Step 6 — roto with SAM 2.

Prompts come from the UI and live in roto/objects.json:

    {"objects": [
        {"id": 1, "name": "hand", "color": "#f04e98",
         "points": {"12": [[812.0, 401.5, 1], [790.0, 380.0, 0]]}}   # frame -> [x, y, label]
    ]}

Coordinates are pixels in the *plate* (the undistorted frames).  Label 1 =
this is the object, 0 = this is not.  Add points on any frame; SAM 2 conditions
on all prompted frames, then we propagate forward from the first prompted
frame and backward from it, so corrections on frame 212 fix frames 213+ and
also influence earlier frames on the next run.

Output: roto/masks/<object id>/000001.png (8-bit, 255 = inside) and
roto/summary.json with per-frame coverage for the timeline strip.

Models download on first use to <vfxpipe folder>/models.  tiny/small run fine on
CPU (~1 fps at 1080p); base_plus is the GPU default for a 3060 Ti.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from ..runner import Step, Context, register, StepError

MODELS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{}"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"   # <vfxpipe folder>/models


def plate_dir(project) -> Path:
    """Frames every step after tracking works on."""
    u = project.dir("track") / "undistorted" / "images"
    return u if u.exists() and any(u.iterdir()) else project.dir("frames")


def objects_path(project) -> Path:
    return project.dir("roto") / "objects.json"


def load_objects(project) -> dict:
    p = objects_path(project)
    if p.exists():
        return json.loads(p.read_text())
    return {"objects": []}


@register
class Roto(Step):
    name = "roto"
    label = "Roto"
    depends = ["track"]
    params = ["sam2_model", "device"]
    outputs = ["masks", "summary.json"]

    def extra_hash(self, project):
        p = objects_path(project)
        return p.read_text() if p.exists() else ""

    def preflight(self, project):
        probs = []
        try:
            import sam2  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            probs.append("sam2 not installed:  pip install git+https://github.com/facebookresearch/sam2.git")
        objs = load_objects(project)["objects"]
        if not any(o.get("points") for o in objs):
            probs.append("no roto points yet — add an object in the Roto view and click on it in a frame")
        if project.settings.get("frame_format", "jpg") != "jpg":
            probs.append("SAM 2 reads JPEG frame folders — set frame_format to jpg")
        return probs

    def run(self, ctx: Context):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        device = ctx.project.device()
        want = ctx.settings.get("sam2_model", "auto")
        if want == "auto":
            want = "base_plus" if device == "cuda" else "tiny"
        ckpt_name, cfg = MODELS[want]
        ckpt = ensure_model(ckpt_name, ctx)
        ctx.log(f"SAM 2 {want} on {device}")

        frames = plate_dir(ctx.project)
        n_frames = len(sorted(frames.glob("*.jpg")))
        objs = load_objects(ctx.project)["objects"]
        objs = [o for o in objs if o.get("points")]

        masks_dir = ctx.out / "masks"
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
        masks_dir.mkdir()
        for o in objs:
            (masks_dir / str(o["id"])).mkdir()

        predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
        ctx.set_progress(0.02, "loading frames")
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True,
                                         offload_state_to_cpu=(device == "cpu"), async_loading_frames=False)
            first = n_frames
            for o in objs:
                for f, pts in sorted(o["points"].items(), key=lambda kv: int(kv[0])):
                    fi = int(f) - 1  # UI frames are 1-based
                    first = min(first, fi)
                    arr = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
                    lab = np.array([int(p[2]) for p in pts], dtype=np.int32)
                    predictor.add_new_points_or_box(state, frame_idx=fi, obj_id=int(o["id"]), points=arr, labels=lab)
            summary = {str(o["id"]): [0.0] * n_frames for o in objs}
            done = 0

            def handle(fi, obj_ids, logits):
                nonlocal done
                for oid, m in zip(obj_ids, logits):
                    mask = (m[0] > 0.0).cpu().numpy()
                    from PIL import Image
                    Image.fromarray((mask * 255).astype(np.uint8), "L").save(masks_dir / str(oid) / f"{fi + 1:06d}.png")
                    summary[str(oid)][fi] = round(float(mask.mean()), 4)
                done += 1
                ctx.set_progress(0.05 + 0.9 * done / max(1, n_frames), f"frame {fi + 1}/{n_frames}")
                ctx.check_cancel()

            for fi, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=first):
                handle(fi, obj_ids, logits)
            if first > 0:
                for fi, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=True):
                    if fi == first:
                        continue
                    handle(fi, obj_ids, logits)
        (ctx.out / "summary.json").write_text(json.dumps({"objects": [
            {"id": o["id"], "name": o.get("name"), "color": o.get("color"), "coverage": summary[str(o["id"])]}
            for o in objs]}))
        return {"objects": len(objs), "frames": n_frames, "model": want}


def ensure_model(name: str, ctx: Context) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p = MODEL_DIR / name
    if not p.exists():
        url = MODEL_URL.format(name)
        ctx.log("downloading", url)
        ctx.set_progress(0.01, f"downloading {name}")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)
    return p
