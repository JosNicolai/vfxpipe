"""Step runner.

Each step declares `depends` (other step names), `params` (which settings keys
affect it) and `outputs` (files it produces).  A step's input hash is built from
its params plus the output hashes of the steps it depends on, so changing the
trim range invalidates everything downstream, but changing the SAM 2 model only
invalidates roto.  Same idea as ComfyUI's partial re-execution, without the graph
editor.

Status lives in  <project>/cache/<step>.json  and logs in <project>/logs/<step>.log
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .project import Project, tool_env

REGISTRY: dict[str, "Step"] = {}
VERBOSE = False   # mirror every log line to the terminal (serve/run --verbose)


def console(*parts: Any) -> None:
    print(*parts, flush=True)
ORDER = ["frames", "track", "splat", "mesh", "lighting", "roto", "export"]


class StepError(RuntimeError):
    pass


class Context:
    """What a running step gets: the project, a logger, a progress setter and a
    cancel flag."""

    def __init__(self, project: Project, step: "Step"):
        self.project = project
        self.step = step
        self.settings = project.settings
        self.out = project.dir(step.name)
        self.log_path = project.dir("logs") / f"{step.name}.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.progress = 0.0
        self.message = ""
        self.cancelled = False

    def log(self, *parts: Any) -> None:
        line = " ".join(str(p) for p in parts)
        stamp = time.strftime("%H:%M:%S")
        self._log.write(f"[{stamp}] {line}\n")
        self._log.flush()
        if VERBOSE:
            console(f"[{self.step.name}] {line}")

    def set_progress(self, fraction: float, message: str = "") -> None:
        self.progress = max(0.0, min(1.0, float(fraction)))
        if message:
            self.message = message

    def check_cancel(self) -> None:
        if self.cancelled:
            raise StepError("cancelled")

    def run(self, cmd: list[str], cwd: Path | None = None, progress_parser: Callable[[str], float | None] | None = None) -> None:
        """Run a subprocess, stream its output to the log, raise on failure."""
        self.log("$", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=tool_env(str(cmd[0])),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
            )
        except FileNotFoundError:
            raise StepError(f"could not launch {cmd[0]} — check the path with:  python -m vfxpipe tools")
        except OSError as e:
            raise StepError(f"could not launch {cmd[0]}: {e}")
        assert proc.stdout is not None
        n_lines = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            n_lines += 1
            self._log.write(line + "\n")
            self._log.flush()
            if VERBOSE:
                console(f"[{self.step.name}]   {line}")
            if progress_parser:
                p = progress_parser(line)
                if p is not None:
                    self.set_progress(p)
            if self.cancelled:
                proc.kill()
                raise StepError("cancelled")
        proc.wait()
        self.log(f"exit code {proc.returncode} ({n_lines} lines of output)")
        if proc.returncode != 0:
            why = ""
            if n_lines == 0:
                why = (" — it produced no output at all, which usually means it never started properly: "
                       "a missing DLL, a .bat wrapper tripping over '&' or spaces in the path, or an "
                       "antivirus block. Try running the same command in a terminal.")
            raise StepError(f"{Path(cmd[0]).name} exited with code {proc.returncode}{why} — see logs/{self.step.name}.log")

    def close(self) -> None:
        self._log.close()


class Step:
    name: str = ""
    label: str = ""
    depends: list[str] = []
    params: list[str] = []
    outputs: list[str] = []          # relative to the step's output dir

    def run(self, ctx: Context) -> dict[str, Any]:
        raise NotImplementedError

    def preflight(self, project: Project) -> list[str]:
        """Return a list of problems that would prevent the step from running."""
        return []

    # ---- hashing --------------------------------------------------------
    def input_hash(self, project: Project) -> str:
        h = hashlib.sha1()
        for key in self.params:
            h.update(f"{key}={json.dumps(project.settings.get(key), sort_keys=True)}\n".encode())
        for dep in self.depends:
            st = load_status(project, dep)
            h.update(f"{dep}:{st.get('output_hash', '')}\n".encode())
        h.update(self.extra_hash(project).encode())
        return h.hexdigest()[:16]

    def extra_hash(self, project: Project) -> str:
        """Hook for steps whose inputs aren't fully captured by settings (roto prompts)."""
        return ""

    def output_hash(self, project: Project) -> str:
        h = hashlib.sha1()
        out = project.dir(self.name)
        for rel in self.outputs:
            p = out / rel
            if p.is_dir():
                files = sorted(p.rglob("*"))
                h.update(f"{rel}:{len(files)}\n".encode())
                for f in files[:2000]:
                    if f.is_file():
                        s = f.stat()
                        h.update(f"{f.name}:{s.st_size}:{int(s.st_mtime)}\n".encode())
            elif p.exists():
                s = p.stat()
                h.update(f"{rel}:{s.st_size}:{int(s.st_mtime)}\n".encode())
            else:
                h.update(f"{rel}:missing\n".encode())
        return h.hexdigest()[:16]


def register(step_cls: type[Step]) -> type[Step]:
    inst = step_cls()
    REGISTRY[inst.name] = inst
    return step_cls


# ---- status persistence ---------------------------------------------------

def status_path(project: Project, name: str) -> Path:
    return project.dir("cache") / f"{name}.json"


def load_status(project: Project, name: str) -> dict[str, Any]:
    p = status_path(project, name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"state": "never", "input_hash": "", "output_hash": ""}


def save_status(project: Project, name: str, st: dict[str, Any]) -> None:
    status_path(project, name).write_text(json.dumps(st, indent=2))


def step_state(project: Project, name: str) -> dict[str, Any]:
    """Status enriched with whether it's stale relative to current inputs."""
    step = REGISTRY[name]
    st = load_status(project, name)
    current = step.input_hash(project)
    st = dict(st)
    st["name"] = name
    st["label"] = step.label
    st["depends"] = step.depends
    st["stale"] = st.get("state") == "done" and st.get("input_hash") != current
    deps_ok = all(load_status(project, d).get("state") == "done" for d in step.depends)
    st["deps_ok"] = deps_ok
    st["problems"] = step.preflight(project)
    return st


# ---- execution --------------------------------------------------------------

class Runner:
    """Runs steps one at a time on a background thread."""

    def __init__(self, project: Project):
        self.project = project
        self.current: Context | None = None
        self.thread: threading.Thread | None = None
        self.queue: list[str] = []
        self.lock = threading.Lock()

    def is_busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def run_steps(self, names: list[str], force: bool = False) -> None:
        with self.lock:
            if self.is_busy():
                raise StepError("a step is already running")
            self.queue = [n for n in ORDER if n in names]
            self.thread = threading.Thread(target=self._worker, args=(force,), daemon=True)
            self.thread.start()

    def cancel(self) -> None:
        if self.current:
            self.current.cancelled = True
        self.queue = []

    def _worker(self, force: bool) -> None:
        while self.queue:
            name = self.queue.pop(0)
            step = REGISTRY[name]
            if not force:
                st = load_status(self.project, name)
                if st.get("state") == "done" and st.get("input_hash") == step.input_hash(self.project):
                    continue  # cached
            problems = step.preflight(self.project)
            if problems:
                save_status(self.project, name, {**load_status(self.project, name), "state": "error",
                                                 "error": "; ".join(problems)})
                console(f"[{name}] cannot run: " + "; ".join(problems))
                self.queue = []
                return
            ctx = Context(self.project, step)
            self.current = ctx
            started = time.time()
            save_status(self.project, name, {"state": "running", "started": started,
                                             "input_hash": step.input_hash(self.project), "output_hash": ""})
            try:
                ctx.log(f"== {name} ==")
                console(f"[{name}] started")
                result = step.run(ctx) or {}
                save_status(self.project, name, {
                    "state": "done", "started": started, "finished": time.time(),
                    "input_hash": step.input_hash(self.project),
                    "output_hash": step.output_hash(self.project),
                    "result": result,
                })
                ctx.log(f"== {name} done in {time.time() - started:.1f}s ==")
                console(f"[{name}] done in {time.time() - started:.1f}s")
            except Exception as exc:  # noqa: BLE001
                ctx.log("ERROR:", exc)
                ctx.log(traceback.format_exc())
                console(f"[{name}] ERROR: {exc}")
                if not isinstance(exc, StepError):
                    console(traceback.format_exc())
                console(f"[{name}] full log: {ctx.log_path}")
                save_status(self.project, name, {
                    "state": "cancelled" if str(exc) == "cancelled" else "error",
                    "started": started, "finished": time.time(),
                    "input_hash": "", "output_hash": "", "error": str(exc),
                })
                self.queue = []
            finally:
                ctx.close()
                self.current = None

    def live(self) -> dict[str, Any]:
        c = self.current
        if not c:
            return {"running": None, "queue": list(self.queue)}
        return {"running": c.step.name, "progress": c.progress, "message": c.message,
                "queue": list(self.queue)}
