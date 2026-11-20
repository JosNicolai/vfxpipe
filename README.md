# vfxpipe

Footage in, Blender scene out — with a browser interface where you can scrub every
frame and check each step's result against the plate.

```
frames ─► track ─► splat ─► mesh ─► lighting ─► export
                 └──────────────────► roto ─────┘
```

| Step | What it does | Runs on | Tools |
|---|---|---|---|
| Frames | ffmpeg extracts a trimmed, optionally downscaled JPEG sequence | CPU | ffmpeg |
| Track | COLMAP features + sequential matching, GLOMAP solve, undistorted plates, floor levelling | CUDA or CPU | COLMAP, GLOMAP |
| Splat | Gaussian splat of the room from the tracked frames | any GPU (Brush); CPU only via a self-built OpenSplat | Brush |
| Mesh | Fits floor/wall/table planes to the points (or a Poisson surface) — shadow catchers with the footage projected onto them | CPU | Open3D (pip) |
| Lighting | Environment map rendered from inside the splat at the character position; grey/chrome ball lookdev | CPU (+ Blender for lookdev) | — |
| Roto | SAM 2 mattes from your clicks, propagated through the shot | CUDA or CPU | SAM 2 (pip) |
| Export | `export/bundle/` of plain files (camera.json, planes.obj, points.ply, env.hdr, mattes, import script) + `scene.blend` when Blender is found | — | (Blender, optional) |

Every step caches its inputs. Change the trim range and everything downstream is
marked stale; change the SAM 2 model and only Roto is.

## Install (Windows)

1. Python 3.10+ from python.org (tick *Add to PATH*).
2. In PowerShell, inside this folder:
   ```
   Set-ExecutionPolicy -Scope Process Bypass
   .\install.ps1            # CUDA torch for the 3060 Ti
   .\install.ps1 -Cpu       # or CPU-only torch
   ```
3. External tools — nothing is installed for you; use what you already have and unzip the rest anywhere.
   Then tell the pipeline where they are, once per machine (skip any that are already on PATH):
   ```
   .\.venv\Scripts\Activate.ps1
   python -m vfxpipe tools --colmap "C:\tools\COLMAP\COLMAP.bat" --glomap "C:\tools\glomap\glomap.exe" --brush "C:\tools\brush\brush_app.exe"
   python -m vfxpipe tools          # shows what is found
   ```
   This writes `tools.json` next to `run.bat`, so the folder stays self-contained (SAM 2 models
   download into `models\` there too). Lookup order: the shot's `project.json` → that `tools.json`
   → `~\.vfxpipe\tools.json` (`--user`) → PATH.
   - ffmpeg: https://www.gyan.dev/ffmpeg/builds/
   - COLMAP (CUDA build): https://github.com/colmap/colmap/releases — point `tools.colmap` at `COLMAP.bat`
   - GLOMAP: https://github.com/colmap/glomap/releases
   - Brush: https://github.com/ArthurBrussee/brush/releases — `brush-app-x86_64-pc-windows-msvc.zip`, prebuilt,
     runs on any GPU. (OpenSplat is also supported but ships no Windows binary; you would have to build it.)
   - Blender 4.2 LTS or newer — an existing install is found automatically (PATH or
     `C:\Program Files\Blender Foundation\`); only set `tools.blender` for a portable copy elsewhere

## Use

```
.\run.bat
```
opens http://127.0.0.1:8765 with the shot picker: create a new shot (name + footage
picked with the file browser) or open an existing one. New shots are created in
`vfxpipe\shots\<name>`; *Open another folder…* opens a shot stored anywhere else.
The *Shots* button in the top-left switches between them at any time.

You can still skip the picker and go straight to a shot:
```
.\run.bat "D:\shots\shot01" "D:\footage\take03.mp4"
```
(PowerShell needs the `.\`; the quotes matter when a path has spaces or `&`.)

Headless, without the interface:
```
python -m vfxpipe run "D:\shots\shot01" --video "D:\footage\take03.mp4" --steps frames,track
python -m vfxpipe run "D:\shots\shot01" --steps all --device cpu
python -m vfxpipe status "D:\shots\shot01"
```

### In the interface

- Left: the current shot and the chain of steps with their state. *Run everything* runs all steps except
  Roto (which needs your clicks first). Each step's panel has *Run* and *Run up to here*.
- Centre: the current frame. Each step draws its own overlay:
  - **Track** — circles are the tracked features, dots are where the solved 3D point
    lands, the line between them is the error. Green/amber/red by size. Use
    *Set character position* here: click a feature where the animal will stand.
  - **Splat** — splat centres projected through the solved camera. Sparse areas will
    light and occlude badly; shoot a walkaround there.
  - **Mesh** — wireframe of the proxy. It should hug the plate on every frame; if it
    slides, the track is bad on those frames.
  - **Lighting** — the lookdev render (grey + chrome ball) or the panorama.
  - **Roto** — click on the object to add a point, shift-click for "not this",
    right-click a point to remove it. Add points on any frames, run, scrub, correct,
    run again. Mattes are tinted per object.
- Bottom: the per-frame strip — solved / high-error / unsolved for the track,
  coverage per object for roto. Drag it, use ← → (shift for 10), space to play.
- Right: settings for the selected step, results, and the live log.

### Getting it into Blender

Two routes. **Bundle** (no Blender needed at export time): `export/bundle/` holds
`camera.json`, `planes.obj`, `points.ply`, `env.hdr`, `masks/` and `import_bundle.py`.
In Blender open the Scripting tab, *Open* `import_bundle.py`, *Run Script*: animated
camera with plate background, planes as shadow catchers with the plate projected on,
points, world, character anchor. Or import the .obj/.ply/.hdr by hand via File → Import.

**Scene**: if Blender is found the step also writes `export/scene.blend`.
Open `export/scene.blend`. Everything is under `TrackRoot`. `CharacterAnchor` is
where you set the character position; put the animal there. The `EnvironmentPlanes`
collection holds one object per fitted plane (`floor_1`, `wall_2`, …), each a Cycles
shadow catcher with the plate projected onto it from `ShotCamera` (UV Project
modifier + emission material). Add boxes/spheres for furniture by hand against
`TrackPoints`, then copy the `PlateProjection` modifier and material onto them.
For the couch, model the cushion as a box, project onto it, and deform that. The compositor already has the plate under the render, and one
image node per roto matte.

**Scale**: the solve has arbitrary units. Measure something (a door ≈ 2.0 m) in
Blender, set `world_scale` in Track to the ratio, re-run Export. Fur, cloth and
physics all need real-world scale.

**Orientation**: the floor is levelled from the average camera "up". If the camera
was tilted the whole take, rotate `TrackRoot`.

**Lighting**: the generated `env.hdr` has the right colour and direction but the
footage's clipped highlights stay clipped (the `env_highlight_boost` setting fakes
some of that back). A real bracketed HDRI dropped in as `lighting/env.hdr` wins.

## Project folder

```
shot01/
  project.json        all settings — edit by hand or via the UI
  frames/             raw extracted frames
  track/              database.db, sparse/, undistorted/{images,sparse}, cameras.json, points.json, points.ply
  splat/              splat.ply, splat_preview.json
  mesh/               mesh.ply, mesh_preview.json
  lighting/           env.hdr, env_preview.png, faces/, lookdev.png
  roto/               objects.json (your clicks), masks/<id>/000001.png, summary.json
  export/             scene.blend
  cache/  logs/       step status + logs
```

## What was tested and what wasn't

Tested on Linux without a GPU, COLMAP or Blender: frame extraction, trim/downscale,
caching and stale detection, the server and UI endpoints, the COLMAP text-model
parser and `cameras.json` export (on a synthetic solve), Poisson mesh, the cubemap
→ equirect → `.hdr` writer (axes verified against Blender's convention), and the
Blender scripts compile.

Not run here: the actual COLMAP/GLOMAP/OpenSplat/Brush/SAM 2/Blender invocations.
Their command lines are written from their documentation; if one has drifted,
`logs/<step>.log` shows the exact command that failed. The Splat step has a `custom`
backend where you can paste your own command with `{input} {output} {iters}`.

## Known limits

- Roto needs JPEG frames (SAM 2 reads a folder of `.jpg`).
- Brush needs a GPU (any vendor). A CPU-only splat means building OpenSplat from source; 2000 iterations is enough for a lighting/occlusion splat.
- The proxy mesh is a shadow catcher, not a hero asset. Expect to clean it up around the parts the character touches.
- Frames GLOMAP can't solve are interpolated linearly in Blender — check the strip.
