"""Runs inside Blender:  blender -b --python build_scene.py -- <project root>

Builds export/scene.blend with:
  * an animated camera from cameras.json, already converted from COLMAP's
    Y-down/Z-forward convention to Blender's, rotated so the estimated floor
    is level (auto_orient) and scaled by world_scale
  * the undistorted plate as camera background + compositor input
  * the proxy mesh as a Cycles shadow catcher, the sparse cloud for reference
  * lighting/env.hdr as the world
  * roto mattes as image-sequence nodes in the compositor (unwired, ready to use)

Everything sits under an empty called TrackRoot: if the auto orientation is
slightly off, rotate/scale that one object and everything follows.
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

root = Path(sys.argv[sys.argv.index("--") + 1])
settings = json.loads((root / "project.json").read_text())
cams = json.loads((root / "track" / "cameras.json").read_text())
frames_meta = json.loads((root / "frames" / "frames.json").read_text())
X = cams["transform"]
if (root / "mesh" / "orientation.json").exists():           # refined from the fitted planes
    X = json.loads((root / "mesh" / "orientation.json").read_text())
M = Matrix(X["M"])
centre = Vector(X["centre"])
scale = float(X["scale"])


def to_world(p):
    """COLMAP world → Blender world (Z-up, centred, scaled)."""
    return (M @ (Vector(p) - centre)) * scale


# ---- fresh scene -------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
for engine in (settings.get("render_engine", "CYCLES"), "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"  # no easing across unsolved gaps
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"   # plate passes through unchanged
scene.render.fps = int(round(frames_meta.get("fps", 25)))
plate_dir = root / "track" / "undistorted" / "images"
if not plate_dir.exists():
    plate_dir = root / "frames"
plates = sorted(plate_dir.glob("*." + settings.get("frame_format", "jpg")))
n = cams["n_frames"]
scene.frame_start, scene.frame_end = 1, n
solved = [f for f in cams["frames"] if f.get("solved")]
if not solved:
    raise SystemExit("no solved frames")
W, H = solved[0]["w"], solved[0]["h"]
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
if scene.render.engine == "CYCLES":
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    try:
        scene.view_layers[0].cycles.use_pass_shadow_catcher = True
    except AttributeError:
        pass

# ---- track root ----------------------------------------------------------------
track_root = bpy.data.objects.new("TrackRoot", None)
scene.collection.objects.link(track_root)

# ---- camera ------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("ShotCamera")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam = bpy.data.objects.new("ShotCamera", cam_data)
cam.parent = track_root
scene.collection.objects.link(cam)
scene.camera = cam
flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))  # COLMAP cam (+Z fwd, -Y up) → Blender cam (-Z fwd, +Y up)
for f in cams["frames"]:
    if not f.get("solved"):
        continue
    fx, fy, cx, cy = f["K"]
    R = Matrix(f["R"])
    rot = (M @ R.transposed() @ flip).to_4x4()
    C = to_world(f["C"])
    mat = Matrix.Translation(C) @ rot
    cam.matrix_world = mat
    cam.keyframe_insert("location", frame=f["frame"])
    cam.keyframe_insert("rotation_euler", frame=f["frame"])
    cam_data.lens = fx / W * 36.0
    cam_data.shift_x = (W / 2 - cx) / max(W, H)
    cam_data.shift_y = (cy - H / 2) / max(W, H)
    cam_data.keyframe_insert("lens", frame=f["frame"])
cam_data.clip_start = 0.01 * scale
cam_data.clip_end = 1000.0 * scale

# ---- plate as background + compositor ---------------------------------------------------
if plates:
    img = bpy.data.images.load(str(plates[0]))
    img.source = "SEQUENCE"
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.image = img
    bg.image_user.frame_duration = n
    bg.image_user.frame_start = 1
    bg.image_user.use_auto_refresh = True
    bg.alpha = 1.0
    bg.display_depth = "BACK"

    if bpy.app.version >= (5, 0, 0):
        nt = bpy.data.node_groups.new("ShotComp", "CompositorNodeTree")
        scene.compositing_node_group = nt
        nt.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        comp = nt.nodes.new("NodeGroupOutput"); comp.location = (700, 100)
    else:
        scene.use_nodes = True
        nt = scene.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        comp = nt.nodes.new("CompositorNodeComposite"); comp.location = (700, 100)
    rl = nt.nodes.new("CompositorNodeRLayers"); rl.location = (0, 300)
    plate = nt.nodes.new("CompositorNodeImage"); plate.image = img; plate.location = (0, -100)
    plate.frame_duration = n; plate.frame_start = 1; plate.use_auto_refresh = True
    plate.label = "Plate (undistorted)"
    over = nt.nodes.new("CompositorNodeAlphaOver"); over.location = (400, 100)
    view = nt.nodes.new("CompositorNodeViewer"); view.location = (700, -100)
    nt.links.new(plate.outputs["Image"], over.inputs[1])
    nt.links.new(rl.outputs["Image"], over.inputs[2])
    nt.links.new(over.outputs["Image"], comp.inputs[0])
    nt.links.new(over.outputs["Image"], view.inputs["Image"])
    # roto mattes, one node each, ready to plug into a Mix or a second Alpha Over
    y = -350
    summary = root / "roto" / "summary.json"
    if summary.exists():
        for o in json.loads(summary.read_text())["objects"]:
            mdir = root / "roto" / "masks" / str(o["id"])
            files = sorted(mdir.glob("*.png"))
            if not files:
                continue
            mimg = bpy.data.images.load(str(files[0]))
            mimg.source = "SEQUENCE"
            node = nt.nodes.new("CompositorNodeImage"); node.image = mimg; node.location = (0, y)
            node.frame_duration = n; node.frame_start = 1; node.use_auto_refresh = True
            node.label = f"Matte: {o.get('name') or o['id']}"
            y -= 250

# ---- projected-footage material (UV Project from the shot camera) ---------------------------
def projected_material(name):
    """Plate projected through ShotCamera.  Non-camera rays see the plate as
    emission (bounce light + reflections on the character are the real room);
    camera rays see white diffuse, which the shadow catcher needs to measure
    shadows."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    emit = nt.nodes.new("ShaderNodeEmission"); emit.location = (350, 0)
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.location = (0, 0)
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.location = (-250, 0); uv.uv_map = "Projected"
    if plates:
        tex.image = bpy.data.images.get(Path(plates[0]).name) or bpy.data.images.load(str(plates[0]))
        tex.image.source = "SEQUENCE"
        tex.image_user.frame_duration = n; tex.image_user.frame_start = 1; tex.image_user.use_auto_refresh = True
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    emit_out = emit.outputs["Emission"]
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (800, 0)
    lp = nt.nodes.new("ShaderNodeLightPath"); lp.location = (300, 250)
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse"); diff.location = (350, -150)
    diff.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    mix = nt.nodes.new("ShaderNodeMixShader"); mix.location = (600, 0)
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(emit_out, mix.inputs[1])            # what the character sees: the plate
    nt.links.new(diff.outputs["BSDF"], mix.inputs[2])  # what the shadow catcher measures
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def add_projection(ob):
    if "Projected" not in ob.data.uv_layers:
        ob.data.uv_layers.new(name="Projected")
    mod = ob.modifiers.new("PlateProjection", "UV_PROJECT")
    mod.uv_layer = "Projected"
    mod.aspect_x = W; mod.aspect_y = H
    mod.projector_count = 1
    mod.projectors[0].object = cam
    ob.data.materials.append(projected_material(f"Plate_{ob.name}"))
    ob.is_shadow_catcher = True
    ob.visible_camera = True


# ---- fitted planes: one object each -----------------------------------------------------------
planes_json = root / "mesh" / "planes.json"
if planes_json.exists():
    coll = bpy.data.collections.new("EnvironmentPlanes")
    scene.collection.children.link(coll)
    for pl in json.loads(planes_json.read_text())["planes"]:
        me = bpy.data.meshes.new(pl["name"])
        me.from_pydata([tuple(to_world(c)) for c in pl["corners"]], [], [(0, 1, 2, 3)])
        me.update()
        ob = bpy.data.objects.new(pl["name"], me)
        ob.parent = track_root
        coll.objects.link(ob)
        add_projection(ob)

# ---- proxy mesh (Poisson, shadow catcher) ---------------------------------------------------
mesh_ply = root / "mesh" / "mesh.ply"
if mesh_ply.exists() and not planes_json.exists():
    bpy.ops.wm.ply_import(filepath=str(mesh_ply))
    ob = bpy.context.view_layer.objects.active or bpy.context.selected_objects[0]
    ob.name = "EnvironmentProxy"
    ob.parent = track_root
    ob.matrix_world = Matrix.Scale(scale, 4) @ M.to_4x4() @ Matrix.Translation(-centre)
    add_projection(ob)

# ---- sparse cloud for reference ------------------------------------------------------------
pts_ply = root / "track" / "points.ply"
if pts_ply.exists():
    bpy.ops.wm.ply_import(filepath=str(pts_ply))
    ob = bpy.context.view_layer.objects.active or bpy.context.selected_objects[0]
    ob.name = "TrackPoints"
    ob.parent = track_root
    ob.matrix_world = Matrix.Scale(scale, 4) @ M.to_4x4() @ Matrix.Translation(-centre)
    ob.hide_render = True
    ob.display_type = "WIRE"

# ---- character anchor + world -------------------------------------------------------------
light_json = root / "lighting" / "lighting.json"
anchor = bpy.data.objects.new("CharacterAnchor", None)
anchor.empty_display_type = "ARROWS"
anchor.parent = track_root
scene.collection.objects.link(anchor)
if light_json.exists():
    pos = json.loads(light_json.read_text())["env_position"]
    anchor.location = to_world(pos)

env = root / "lighting" / "env.hdr"
world = bpy.data.worlds.new("ShotWorld")
scene.world = world
world.use_nodes = True
wn = world.node_tree
if env.exists():
    tex = wn.nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.load(str(env))
    tex.location = (-400, 0)
    wn.links.new(tex.outputs["Color"], wn.nodes["Background"].inputs["Color"])
else:
    wn.nodes["Background"].inputs["Color"].default_value = (0.3, 0.3, 0.3, 1)

# ---- device ----------------------------------------------------------------------------------
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    (prefs.refresh_devices if hasattr(prefs, "refresh_devices") else prefs.get_devices)()
    if any(d.type == "CUDA" for d in prefs.devices):
        for d in prefs.devices:
            d.use = d.type == "CUDA"
        scene.cycles.device = "GPU"
except Exception:
    pass

out = root / "export" / "scene.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(out))
print("saved", out)
