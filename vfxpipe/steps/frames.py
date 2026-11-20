"""Step 1 — extract an image sequence from the footage with ffmpeg."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from ..runner import Step, Context, register, StepError, cloud_hint


def probe(ffprobe: str | None, ffmpeg: str, video: str) -> dict:
    """Video size / fps / frame count.  Uses ffprobe when available, else
    parses ffmpeg's own stream summary (no frame count in that case)."""
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
               "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,nb_frames",
               "-of", "json", video]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except FileNotFoundError:
            raise StepError(f"could not launch ffprobe at {ffprobe}")
        except OSError as e:
            raise StepError(f"could not launch ffprobe at {ffprobe}: {e}{cloud_hint(e)}")
        except subprocess.CalledProcessError as e:
            raise StepError(f"ffprobe could not read the footage: {e.stderr.strip()[-300:]}")
        s = json.loads(out)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {"width": s["width"], "height": s["height"], "fps": float(num) / float(den),
                "frames": int(s.get("nb_read_packets") or s.get("nb_frames") or 0)}
    try:
        info = subprocess.run([ffmpeg, "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    except FileNotFoundError:
        raise StepError(f"could not launch ffmpeg at {ffmpeg}")
    except OSError as e:
        raise StepError(f"could not launch ffmpeg at {ffmpeg}: {e}{cloud_hint(e)}")
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    f = re.search(r"([\d.]+) fps", info)
    if not m:
        raise StepError("ffmpeg could not read the footage:\n" + info.strip()[-300:])
    return {"width": int(m.group(1)), "height": int(m.group(2)), "fps": float(f.group(1)) if f else 25.0, "frames": 0}


def _supports_fps_mode(ffmpeg: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=20).stdout
        return "-fps_mode" in out
    except Exception:  # noqa: BLE001
        return True


@register
class Frames(Step):
    name = "frames"
    label = "Frames"
    depends = []
    params = ["source_video", "in_frame", "out_frame", "max_width", "frame_format", "jpg_quality"]
    outputs = ["."]

    def extra_hash(self, project):
        p = Path(project.settings.get("source_video") or "")
        return f"{p.stat().st_size}:{int(p.stat().st_mtime)}" if p.exists() else "nofile"

    def preflight(self, project):
        probs = []
        if not project.tool("ffmpeg"):
            probs.append("ffmpeg not found — install it and add to PATH or set tools.ffmpeg")
        src = project.settings.get("source_video")
        if not src:
            probs.append("no source video set")
        elif not Path(src).exists():
            probs.append(f"source video not found: {src}")
        return probs

    def run(self, ctx: Context):
        s = ctx.settings
        ffmpeg = ctx.project.tool("ffmpeg")
        ffprobe = ctx.project.tool("ffprobe")
        ctx.log("ffmpeg:", ffmpeg, "| ffprobe:", ffprobe or "not found (frame count unknown; progress bar will be approximate)")
        src = s["source_video"]
        info = probe(ffprobe, ffmpeg, src)
        ctx.log("source:", info)
        # clear old frames
        for f in ctx.out.glob("*"):
            if f.is_file():
                f.unlink()
        in_f = int(s.get("in_frame", 0))
        out_f = int(s.get("out_frame", -1))
        total = info["frames"] or 0
        if out_f < 0 or (total and out_f >= total):
            out_f = (total - 1) if total else -1
        vf = [f"select=gte(n\\,{in_f})" + (f"*lte(n\\,{out_f})" if out_f >= 0 else "")]
        mw = int(s.get("max_width", 0) or 0)
        if mw and info["width"] > mw:
            vf.append(f"scale={mw}:-2")  # -2 keeps even height
        ext = s.get("frame_format", "jpg")
        # -fps_mode passthrough replaced -vsync 0 in ffmpeg 5.1+; older builds only know -vsync
        cmd = [ffmpeg, "-hide_banner", "-y", "-i", src, "-vf", ",".join(vf),
               "-fps_mode", "passthrough" if _supports_fps_mode(ffmpeg) else "0"]
        if cmd[-2] == "-fps_mode" and cmd[-1] == "0":
            cmd[-2] = "-vsync"
        if ext == "jpg":
            cmd += ["-q:v", str(s.get("jpg_quality", 3))]
        cmd += [str(ctx.out / f"%06d.{ext}")]
        expected = (out_f - in_f + 1) if out_f >= 0 else 0

        def prog(line: str):
            m = re.search(r"frame=\s*(\d+)", line)
            if m and expected:
                return int(m.group(1)) / expected
            return None

        ctx.set_progress(0.0, "extracting frames")
        ctx.run(cmd, progress_parser=prog)
        n = len(list(ctx.out.glob(f"*.{ext}")))
        if n == 0:
            raise StepError("ffmpeg produced no frames — check in/out range")
        meta = {**info, "extracted": n, "in_frame": in_f, "out_frame": out_f,
                "frame_width": None, "frame_height": None}
        try:
            from PIL import Image
            with Image.open(next(ctx.out.glob(f"*.{ext}"))) as im:
                meta["frame_width"], meta["frame_height"] = im.size
        except Exception as e:  # noqa: BLE001
            ctx.log("could not read frame size:", e)
        (ctx.out / "frames.json").write_text(json.dumps(meta, indent=2))
        ctx.log(f"extracted {n} frames")
        return {"frames": n, "width": meta["frame_width"], "height": meta["frame_height"], "fps": info["fps"]}
