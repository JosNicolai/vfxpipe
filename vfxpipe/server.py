"""Local web server:  python -m vfxpipe serve [shot folder]

Shots live in <vfxpipe folder>/shots/<name> by default.  The interface can
create a new shot, open any existing shot folder, and switch between them.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from . import steps  # noqa: F401  (registers steps)
from .project import Project, APP_ROOT
from .runner import Runner, ORDER, step_state, StepError
from .steps.roto import plate_dir, load_objects, objects_path

WEB = Path(__file__).resolve().parent / "web"
SHOTS_ROOT = APP_ROOT / "shots"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mxf", ".avi", ".m4v", ".webm", ".braw", ".mts", ".m2ts"}
RECENT = APP_ROOT / "recent_shots.json"


class State:
    def __init__(self, initial: str | None):
        self.project: Project | None = None
        self.runner: Runner | None = None
        if initial:
            self.open(initial)

    def open(self, path: str) -> Project:
        if self.runner and self.runner.is_busy():
            raise StepError("a step is running — cancel it before switching shots")
        self.project = Project(path)
        self.runner = Runner(self.project)
        self._remember(self.project.root)
        return self.project

    def _remember(self, root: Path) -> None:
        rec = self.recent()
        rec = [str(root)] + [r for r in rec if r != str(root)]
        RECENT.write_text(json.dumps(rec[:20]))

    def forget(self, path: str) -> None:
        rec = [r for r in self.recent() if r != path]
        RECENT.write_text(json.dumps(rec))

    def close_if_current(self, path: str) -> None:
        if self.project and str(self.project.root) == path:
            self.project, self.runner = None, None

    @staticmethod
    def recent() -> list[str]:
        if RECENT.exists():
            try:
                return [r for r in json.loads(RECENT.read_text()) if Path(r).exists()]
            except json.JSONDecodeError:
                pass
        return []

    def shots(self) -> list[dict[str, Any]]:
        seen, out = set(), []
        SHOTS_ROOT.mkdir(exist_ok=True)
        paths = [str(p) for p in sorted(SHOTS_ROOT.iterdir()) if (p / "project.json").exists()] + self.recent()
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            pj = Path(p) / "project.json"
            if not pj.exists():
                continue
            try:
                s = json.loads(pj.read_text())
            except json.JSONDecodeError:
                s = {}
            frames = len(list((Path(p) / "frames").glob("*.jpg"))) + len(list((Path(p) / "frames").glob("*.png")))
            out.append({"path": p, "name": Path(p).name, "video": s.get("source_video", ""),
                        "frames": frames, "modified": pj.stat().st_mtime,
                        "current": bool(self.project and str(self.project.root) == p)})
        return out


def load_transform_safe(project):
    from .colmap_io import load_transform
    try:
        return load_transform(project)
    except Exception:  # noqa: BLE001
        return None


def make_app(initial: str | None) -> FastAPI:
    app = FastAPI(title="vfxpipe")
    state = State(initial)

    def P() -> Project:
        if not state.project:
            raise HTTPException(409, "no shot open")
        return state.project

    def R() -> Runner:
        P()
        return state.runner  # type: ignore[return-value]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text(encoding="utf-8")

    # ---- shots -----------------------------------------------------------------
    @app.get("/api/shots")
    def shots():
        return {"shots": state.shots(), "root": str(SHOTS_ROOT),
                "current": str(state.project.root) if state.project else None}

    @app.post("/api/shots/open")
    async def shots_open(req: Request):
        body = await req.json()
        path = body.get("path", "")
        if not path or not Path(path).is_dir():
            raise HTTPException(404, "folder not found")
        try:
            state.open(path)
        except StepError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "path": str(state.project.root)}  # type: ignore[union-attr]

    @app.post("/api/shots/new")
    async def shots_new(req: Request):
        body = await req.json()
        name = re.sub(r"[^\w\- ]+", "", body.get("name", "")).strip()
        video = body.get("video", "")
        if not name:
            raise HTTPException(400, "give the shot a name")
        if video and not Path(video).exists():
            raise HTTPException(404, f"footage not found: {video}")
        root = Path(body.get("root") or SHOTS_ROOT) / name
        if (root / "project.json").exists():
            raise HTTPException(409, f"a shot called '{name}' already exists — open it instead")
        try:
            proj = state.open(str(root))
        except StepError as e:
            raise HTTPException(409, str(e))
        if video:
            proj.update({"source_video": video})
        return {"ok": True, "path": str(proj.root)}

    @app.post("/api/shots/delete")
    async def shots_delete(req: Request):
        """delete_files=true removes the whole shot folder; otherwise it is only dropped from the list."""
        import shutil
        body = await req.json()
        path = body.get("path", "")
        target = Path(path)
        if not path or not (target / "project.json").exists():
            raise HTTPException(404, "not a shot folder")
        if state.runner and state.runner.is_busy() and state.project and str(state.project.root) == path:
            raise HTTPException(409, "a step is running on this shot — cancel it first")
        state.close_if_current(path)
        state.forget(path)
        if body.get("delete_files"):
            shutil.rmtree(target)
        return {"ok": True}

    # ---- project ------------------------------------------------------------------
    @app.get("/api/project")
    def get_project():
        if not state.project:
            return {"open": False, "shots": state.shots(), "shots_root": str(SHOTS_ROOT)}
        project = state.project
        st = {n: step_state(project, n) for n in ORDER}
        pd = plate_dir(project)
        return {**project.to_dict(), "open": True, "name": project.root.name,
                "steps": st, "order": ORDER, "live": state.runner.live(),  # type: ignore[union-attr]
                "plate_count": len(list(pd.glob("*." + project.settings.get("frame_format", "jpg")))),
                "plate_is_undistorted": pd.name == "images"}

    @app.post("/api/settings")
    async def set_settings(req: Request):
        body = await req.json()
        P().update(body)
        return {"ok": True, "settings": P().settings}

    @app.post("/api/run")
    async def run(req: Request):
        body = await req.json()
        names = body.get("steps") or ORDER
        try:
            R().run_steps(names, force=bool(body.get("force")))
        except StepError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/cancel")
    def cancel():
        R().cancel()
        return {"ok": True}

    @app.get("/api/log/{step}")
    def log(step: str, tail: int = 300):
        p = P().dir("logs") / f"{step}.log"
        if not p.exists():
            return {"lines": []}
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-tail:]}

    # ---- media -----------------------------------------------------------------------
    @app.get("/api/frame/{n}")
    def frame(n: int, src: str = "plate"):
        project = P()
        ext = project.settings.get("frame_format", "jpg")
        d = plate_dir(project) if src == "plate" else project.dir("frames")
        p = d / f"{n:06d}.{ext}"
        if not p.exists():
            raise HTTPException(404, "no such frame")
        return FileResponse(p, headers={"Cache-Control": "no-cache"})

    @app.get("/api/mask/{obj}/{n}")
    def mask(obj: int, n: int):
        p = P().dir("roto") / "masks" / str(obj) / f"{n:06d}.png"
        if not p.exists():
            raise HTTPException(404, "no mask")
        return FileResponse(p, headers={"Cache-Control": "no-cache"})

    @app.get("/api/transform")
    def transform():
        from .colmap_io import load_transform
        project = P()
        if not (project.dir("track") / "cameras.json").exists():
            raise HTTPException(404, "no track")
        return load_transform(project)

    @app.post("/api/orient/plane")
    async def orient_plane(req: Request):
        """Use a fitted plane's normal as 'up' (manual override), then re-run the plane naming."""
        body = await req.json()
        planes = json.loads((P().dir("mesh") / "planes.json").read_text())["planes"]
        pl = next((x for x in planes if x["name"] == body.get("name")), None)
        if not pl:
            raise HTTPException(404, "no such plane")
        n = pl["normal"]
        cur = load_transform_safe(P())
        if cur is not None and sum(a * b for a, b in zip(n, cur["estimated_up"])) < 0:
            n = [-v for v in n]
        P().update({"up_vector": n})
        return {"ok": True, "up_vector": n}

    @app.get("/api/file/{step}/{path:path}")
    def file(step: str, path: str):
        project = P()
        if step not in ORDER or ".." in path:
            raise HTTPException(400, "bad path")
        p = (project.dir(step) / path).resolve()
        if not str(p).startswith(str(project.dir(step).resolve())) or not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(p, headers={"Cache-Control": "no-cache"})

    # ---- roto prompts -------------------------------------------------------------------
    @app.get("/api/roto/objects")
    def roto_get():
        return load_objects(P())

    @app.put("/api/roto/objects")
    async def roto_put(req: Request):
        body = await req.json()
        objects_path(P()).write_text(json.dumps(body, indent=1))
        return {"ok": True, "stale": step_state(P(), "roto")["stale"]}

    # ---- file browser (footage and folders) ----------------------------------------------
    @app.get("/api/ls")
    def ls(path: str = ""):
        if not path:
            if os.name == "nt":
                import string
                from ctypes import windll
                mask = windll.kernel32.GetLogicalDrives()
                drives = [f"{c}:\\" for i, c in enumerate(string.ascii_uppercase) if mask & (1 << i)]
                return {"path": "", "parent": "", "dirs": drives, "files": []}
            path = "/"
        p = Path(path)
        if not p.is_dir():
            raise HTTPException(404, "not a directory")
        dirs, files = [], []
        try:
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if c.name.startswith(".") or c.name.startswith("$"):
                    continue
                if c.is_dir():
                    dirs.append(c.name)
                elif c.suffix.lower() in VIDEO_EXT:
                    files.append(c.name)
        except PermissionError:
            pass
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "dirs": dirs, "files": files,
                "is_shot": (p / "project.json").exists()}

    return app


def serve(project_root: str | None = None, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    app = make_app(project_root)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"vfxpipe — shots in {SHOTS_ROOT}\n→ http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
