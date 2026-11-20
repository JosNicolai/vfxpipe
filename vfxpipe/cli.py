"""python -m vfxpipe <command>

  serve  [project] [--video FILE] [--port 8765] [--no-browser]   start the UI (pick or create a shot inside it)
  run    <project> [--steps frames,track,...] [--force]           run steps headless
  status <project>                                                 print step states
  tools  [--colmap PATH] [--brush PATH] ...                        set tool paths once (tools.json next to run.bat)
"""
from __future__ import annotations

import argparse
import sys
import time

from . import steps  # noqa: F401
from .project import Project, TOOL_NAMES, MACHINE_TOOLS, load_machine_tools, save_machine_tools
from .runner import Runner, ORDER, step_state, load_status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vfxpipe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("project", nargs="?"); s.add_argument("--video"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true"); s.add_argument("--device", choices=["auto", "cuda", "cpu"])
    s.add_argument("--verbose", "-v", action="store_true", help="mirror full step logs to the terminal")
    r = sub.add_parser("run"); r.add_argument("project"); r.add_argument("--steps", default="all"); r.add_argument("--force", action="store_true")
    r.add_argument("--video"); r.add_argument("--device", choices=["auto", "cuda", "cpu"])
    r.add_argument("--verbose", "-v", action="store_true", help="mirror full step logs to the terminal")
    st = sub.add_parser("status"); st.add_argument("project")
    tl = sub.add_parser("tools", help="set per-machine tool paths, shared by all shots")
    for n in TOOL_NAMES:
        tl.add_argument(f"--{n}", metavar="PATH")
    tl.add_argument("--user", action="store_true", help="write to ~/.vfxpipe/tools.json instead of the vfxpipe folder")
    a = ap.parse_args(argv)
    if getattr(a, "verbose", False):
        from . import runner
        runner.VERBOSE = True

    if a.cmd == "tools":
        given = {n: getattr(a, n) for n in TOOL_NAMES if getattr(a, n)}
        if given:
            import os
            for n, v in given.items():
                if not os.path.exists(v):
                    print(f"warning: {n}: {v} does not exist")
            print("saved to", save_machine_tools(given, user=a.user))
        print(f"lookup: project.json > {MACHINE_TOOLS} > ~/.vfxpipe/tools.json > PATH\n")
        probe = Project.__new__(Project); probe.settings = {"tools": {}}
        for n, v in load_machine_tools().items():
            if any(ord(c) < 32 for c in v):
                print(f"  !! {n}: path contains a control character (a '\\t', '\\n' or '\\b' in the JSON became a tab/newline). "
                      f"Use forward slashes or double backslashes in tools.json.")
        for n in TOOL_NAMES:
            found = probe.tool(n)
            src = "set here" if n in load_machine_tools() else ("on PATH" if found else "")
            print(f"  {n:10s} {found or 'NOT FOUND':60s} {src}")
        return 0

    if a.cmd == "serve" and not a.project:
        from .server import serve
        serve(None, port=a.port, open_browser=not a.no_browser)
        return 0

    project = Project(a.project)
    upd = {}
    if getattr(a, "video", None):
        upd["source_video"] = str(a.video)
    if getattr(a, "device", None):
        upd["device"] = a.device
    if upd:
        project.update(upd)

    if a.cmd == "serve":
        from .server import serve
        serve(str(project.root), port=a.port, open_browser=not a.no_browser)
        return 0

    if a.cmd == "status":
        for n in ORDER:
            s = step_state(project, n)
            flag = "stale" if s["stale"] else s["state"]
            print(f"{n:10s} {flag:10s} {s.get('error', '') or ''}")
        return 0

    names = ORDER if a.steps == "all" else [x.strip() for x in a.steps.split(",")]
    runner = Runner(project)
    runner.run_steps(names, force=a.force)
    last = None
    while runner.is_busy():
        live = runner.live()
        key = (live.get("running"), round(live.get("progress", 0) * 100))
        if key != last and live.get("running"):
            print(f"\r{live['running']:10s} {key[1]:3d}%  {live.get('message', '')[:60]:60s}", end="", flush=True)
            last = key
        time.sleep(0.5)
    print()
    ok = True
    for n in names:
        s = load_status(project, n)
        print(f"{n:10s} {s.get('state')}  {s.get('error', '') or ''}")
        ok &= s.get("state") == "done"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
