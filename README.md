# vfxpipe

Footage in, Blender scene out, with a browser interface where you can scrub every
frame and check each step's result against the plate. Built for putting a CG
character into live-action footage with free / open-source tools.

```
frames -> track -> splat -> mesh -> lighting -> export
                 \-----------------> roto -----/
```

| Step | What it does | Runs on | Tools |
|---|---|---|---|
| Frames | ffmpeg extracts a trimmed, optionally downscaled JPEG sequence | CPU | ffmpeg |
| Track | COLMAP features + sequential matching, GLOMAP solve, undistorted plates, floor levelling | CUDA or CPU | COLMAP, GLOMAP |
| Splat | Gaussian splat of the room from the tracked frames | any GPU (Brush); CPU only via a self-built OpenSplat | Brush |
| Mesh | Proxy geometry to project the footage on: fitted planes (floor / wall / ceiling / table), a Poisson surface, or a COLMAP dense mesh | CPU (dense: CUDA) | Open3D (pip), COLMAP |
| Lighting | Environment map rendered from inside the splat at the character position; grey/chrome ball lookdev | CPU (+ Blender for lookdev) | - |
| Roto | SAM 2 mattes from your clicks, propagated through the shot | CUDA or CPU | SAM 2 (pip) |
| Export | `export/bundle/` of plain files (camera, planes, points, environment map, mattes, import script) + `scene.blend` when Blender is found | - | Blender (optional) |

Every step caches its inputs. Change the trim range and everything downstream is
marked stale; change the SAM 2 model and only Roto is.

## Install (Windows)

**Keep the project, `.venv` and the tools out of OneDrive.** Files-on-demand turns
`.exe` files into online-only placeholders and Windows then fails with
`WinError 389` ("cloud operation was unsuccessful"). Something like `C:\vfxpipe` works well.

1. **Python 3.10 - 3.12** from https://www.python.org/downloads/windows/ (tick *Add python.exe to PATH*).
   The Microsoft Store Python is skipped on purpose: it cannot create virtual environments reliably.
2. In PowerShell, inside this folder, one command at a time:
   ```
   Set-ExecutionPolicy -Scope Process Bypass
   .\install.ps1
   ```
   This creates `.venv`, installs vfxpipe, Open3D, SAM 2 and the right torch build for your GPU
   (read from `nvidia-smi`):

   | GPU | torch build |
   |---|---|
   | GTX 10 / RTX 20 / 30 / 40 | `cu126` |
   | RTX 50-series | `cu128` |
   | no NVIDIA GPU | CPU torch (roto is slower; Track runs on CPU too) |

   Override with `.\install.ps1 -Cuda cu118|cu126|cu128` or `.\install.ps1 -Cpu`.
   The last line printed is `torch ... CUDA available: True/False`; include it when reporting GPU problems.
3. **External tools.** Either let the script fetch them:
   ```
   .\setup-tools.ps1                 # ffmpeg, COLMAP, GLOMAP, Brush
   .\setup-tools.ps1 -NoCuda         # CPU builds of COLMAP / GLOMAP
   .\setup-tools.ps1 -Only ffmpeg,brush
   ```
   They are downloaded from the official releases into `.\tools` (or `%LOCALAPPDATA%\vfxpipe\tools`
   when the project is inside OneDrive) and registered in `tools.json`. If a download fails the script says
   which one and where to get it by hand.

   Or use what you already have and register it once per machine:
   ```
   .\.venv\Scripts\Activate.ps1
   python -m vfxpipe tools --ffmpeg "C:\tools\ffmpeg\bin\ffmpeg.exe" --colmap "C:\tools\colmap\bin\colmap.exe" --glomap "C:\tools\glomap\glomap.exe" --brush "C:\tools\brush\brush_app.exe"
   python -m vfxpipe tools          # shows what is found
   ```
   `tools.json` sits next to `run.bat` (per machine, not committed). Lookup order: the shot's
   `project.json` -> `tools.json` -> `~\.vfxpipe\tools.json` (`--user`) -> PATH. Use forward slashes or
   doubled backslashes when editing it by hand (`\t` in a path is a tab).
   - ffmpeg: https://www.gyan.dev/ffmpeg/builds/
   - COLMAP: https://github.com/colmap/colmap/releases (CUDA build for NVIDIA; `colmap.exe` is used directly, `COLMAP.bat` breaks on paths containing `&`)
   - GLOMAP: https://github.com/colmap/glomap/releases
   - Brush: https://github.com/ArthurBrussee/brush/releases (prebuilt, runs on any GPU). OpenSplat is also supported but ships no Windows binary.
   - Blender 4.2 LTS or newer: an existing install is found automatically (PATH or
     `C:\Program Files\Blender Foundation\`). Otherwise `winget install BlenderFoundation.Blender`;
     set `tools.blender` only for a portable copy. Blender 5.x is supported.

SAM 2 models download into `models\` in this folder on first use.

## Use

```
.\run.bat
```
opens http://127.0.0.1:8765 with the shot picker: **New shot** (name + footage picked with the file
browser) on top, **previous shots** below. New shots are created in `shots\<name>`;
*Open another folder...* opens a shot stored anywhere else. Shots can be forgotten or deleted from the
picker, and the *Shots* button top-left switches between them at any time. Settings and step state are
saved per shot.

Skip the picker and go straight to a shot:
```
.\run.bat "D:\shots\shot01" "D:\footage\take03.mp4"
```
(PowerShell needs the `.\`; the quotes matter when a path has spaces or `&`.)
`set VFXPIPE_VERBOSE=1` before `run.bat` mirrors the full step output into the terminal.

Headless, without the interface:
```
python -m vfxpipe run "D:\shots\shot01" --video "D:\footage\take03.mp4" --steps frames,track
python -m vfxpipe run "D:\shots\shot01" --steps all --device cpu
python -m vfxpipe status "D:\shots\shot01"
```

### In the interface

- Left: the current shot and the chain of steps with their state. *Run everything* runs all steps except
  Roto (which needs your clicks first). Each step's panel has *Run* and *Run up to here*. A pill at the top
  shows what is busy. Hover a setting for what it does and a sensible value.
- Centre: the current frame. Each step draws its own overlay:
  - **Track**: circles are the tracked features, dots are where the solved 3D point lands, the line
    between them is the error (green/amber/red by size). The point cloud and camera path can be shown
    (thinned to 1 in N points). *Set character position*: click where the animal will stand.
  - **Splat**: splat centres projected through the solved camera. Sparse areas will light and occlude
    badly; shoot a walkaround there.
  - **Mesh**: wireframe of the proxy, plus a list of fitted planes (floor / flat / wall / ceiling / slope)
    with *use as floor* if the automatic pick is wrong. It should hug the plate on every frame; if it
    slides, the track is bad on those frames.
  - **Lighting**: the lookdev render (grey + chrome ball) or the panorama.
  - **Roto**: click on the object to add a point, shift-click for "not this", right-click a point to
    remove it. Add points on any frames, run, scrub, correct, run again. Mattes are tinted per object.
  - **3D** button: orbit the solved scene (drag orbit, wheel zoom, shift-drag pan, R reset).
  - *Reload viewer data* re-reads a step's output if an overlay looks empty.
- Bottom: the per-frame strip (solved / high-error / unsolved for the track, coverage per object for
  roto). Drag it, use the arrow keys (shift for 10), space to play.
- Right: settings for the selected step, results, and the live log.

### Mesh modes

`mesh_mode` in the Mesh step: `planes` (default; a handful of fitted planes, the quickest way to get
something to project footage on), `poisson` (surface from the splat or sparse points) or
`colmap_dense` (COLMAP patch-match stereo + Poisson; best geometry, needs CUDA COLMAP, slow). Plane
settings: `plane_max`, `plane_min_inliers`, `plane_tolerance`.

### Getting it into Blender

Two routes. **Bundle** (no Blender needed at export time): `export/bundle/` holds
`camera.json`, `planes.obj` or `proxy.ply`, `points.ply`, `env.hdr`, `masks/` and `import_bundle.py`.
In Blender open the Scripting tab, *Open* `import_bundle.py`, *Run Script*: animated camera with the plate
as background, planes/proxy as shadow catchers with the plate projected on (UV Project), track points,
world light, character anchor, and a compositor setup. Or import the .obj/.ply/.hdr by hand.

**Scene**: if Blender is found the step also writes `export/scene.blend`. Everything is under `TrackRoot`.
`CharacterAnchor` is where you set the character position. `EnvironmentPlanes` holds one object per fitted
plane (`floor_1`, `wall_2`, ...), each a Cycles shadow catcher with the plate projected from `ShotCamera`.
Add boxes for furniture by hand against `TrackPoints`, then copy the `PlateProjection` modifier and
material onto them.

**Making it look real** (Cycles, film transparent, *Standard* view transform):
- The compositor multiplies the plate with the *Shadow Catcher* pass and lays the character over it.
  The catchers themselves do not render the plate; it comes from the compositor. If you see
  "compositor setup failed" in Blender's console, build that one node chain by hand.
- The character is lit by `env.hdr` (linear light). Add a real light for any strong key the footage shows.
- Match grain, lens blur and colour of the plate on the character in the compositor.
- Use `lighting/lookdev.png` (grey + chrome ball) to check exposure and direction before rendering.

**Scale**: the solve has arbitrary units. Measure something (a door is about 2.0 m) in Blender, set
`world_scale` in Track to the ratio, re-run Export. Fur, cloth and physics need real-world scale.

**Orientation**: Track levels the scene on the largest horizontal plane (the floor), not on the average
camera up, so a tilted camera is handled. If it picks the wrong plane (a couch top, say), use *use as floor*
in the Mesh planes list or set `up_vector` by hand, then re-run Export.

**Lighting**: the generated `env.hdr` has the right colour and direction but the footage's clipped
highlights stay clipped (`env_highlight_boost` fakes some back). A real bracketed HDRI dropped in as
`lighting/env.hdr` wins.

## Project folder

```
vfxpipe/                 this repository
  tools.json             per-machine tool paths (not committed)
  shots/                 shots created from the picker (not committed)
  models/                downloaded SAM 2 weights (not committed)
  tools/                 tools downloaded by setup-tools.ps1 (not committed)

shot01/
  project.json           all settings for the shot, edit by hand or via the UI
  frames/                extracted frames
  track/                 database.db, sparse/, undistorted/{images,sparse}, cameras.json, points.json, points.ply
  splat/                 splat.ply, splat_preview.json
  mesh/                  planes / mesh files, orientation.json, mesh_preview.json
  lighting/              env.hdr, env_preview.png, faces/, lookdev.png
  roto/                  objects.json (your clicks), masks/<id>/000001.png, summary.json
  export/                bundle/ and scene.blend
  cache/  logs/          step status + one log per step
```

## Troubleshooting

Every step writes `logs/<step>.log` (with the exact command and its exit code) and prints
`[step] ERROR: ...` with a traceback in the terminal.

| Symptom | Cause / fix |
|---|---|
| `WinError 389` / "cloud operation" | A tool or the `.venv` is an online-only OneDrive file. Keep it on the device or move it out of OneDrive (`setup-tools.ps1`). |
| `WinError 2` / "could not launch" | Wrong path in `tools.json`. `python -m vfxpipe tools` shows what resolves. |
| `install.ps1` hangs at the venv | Microsoft Store Python. Install python.org Python, delete `.venv`, run again. |
| Torch has no matching distribution | Python too new. Use 3.12, or `-Cuda cu126`. |
| Roto runs on CPU | `torch.cuda.is_available()` is False. Re-run `install.ps1` and check the printed GPU line. |
| Floor is diagonal / couch is the floor | See *Orientation* above. |

## What was tested and what wasn't

Tested on Linux without a GPU, COLMAP or Blender: frame extraction, trim/downscale, caching and stale
detection, the server and UI endpoints, the COLMAP text-model parser and `cameras.json` export (on a
synthetic solve), plane fitting, Poisson mesh, the cubemap -> equirect -> `.hdr` writer (axes verified
against Blender's convention), and the Blender scripts compile. On a real Windows machine so far: Frames,
Track, COLMAP dense mesh and the Blender import.

Not independently verified: SAM 2 on every Python / torch combination, Brush command-line flags, the
Blender 5.x compositor node names, `setup-tools.ps1` release-name matching, and the RTX 50-series path.
Command lines are written from each tool's documentation; if one has drifted, `logs/<step>.log` shows the
exact command that failed. The Splat step has a `custom` backend where you can paste your own command with
`{input} {output} {iters}`.

## Known limits

- Roto needs JPEG frames (SAM 2 reads a folder of `.jpg`).
- Brush needs a GPU (any vendor). A CPU-only splat means building OpenSplat from source; 2000 iterations is enough for a lighting/occlusion splat.
- The proxy mesh is a shadow catcher, not a hero asset. Expect to clean it up around the parts the character touches.
- Frames GLOMAP can't solve are interpolated linearly in Blender; check the strip.
