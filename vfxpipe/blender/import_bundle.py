"""Run this inside Blender: Scripting tab → Open → this file → Run Script.
It looks for the bundle files next to itself and sets up the shot:
animated camera + plate background, fitted planes (shadow catchers with the
plate projected on), track points, environment light, character anchor.
Works in Blender 4.2 – 5.x.  Delete what you don't want afterwards."""
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion

try:
    here = Path(__file__).resolve().parent            # Blender sets __file__ when run from the text editor
except NameError:
    here = Path(bpy.context.space_data.text.filepath).resolve().parent
cam = json.loads((here / "camera.json").read_text())
scene = bpy.context.scene
scene.frame_start, scene.frame_end = cam["frame_start"], cam["frame_end"]
scene.render.resolution_x, scene.render.resolution_y = cam["width"], cam["height"]
scene.render.fps = int(round(cam["fps"]))
scene.render.film_transparent = True
scene.render.engine = "CYCLES"
scene.view_settings.view_transform = "Standard"   # plate passes through unchanged
try:
    scene.view_layers[0].cycles.use_pass_shadow_catcher = True
except AttributeError:
    pass

root = bpy.data.objects.new("TrackRoot", None); scene.collection.objects.link(root)
cd = bpy.data.cameras.new("ShotCamera"); cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0
co = bpy.data.objects.new("ShotCamera", cd); co.parent = root; scene.collection.objects.link(co); scene.camera = co
co.rotation_mode = "QUATERNION"
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
for f in cam["frames"]:
    co.location = f["location"]; co.rotation_quaternion = Quaternion(f["quaternion"])
    cd.lens = f["lens_mm"]; cd.shift_x = f["shift_x"]; cd.shift_y = f["shift_y"]
    for ob, keys in ((co, ("location", "rotation_quaternion")), (cd, ("lens", "shift_x", "shift_y"))):
        for k in keys:
            ob.keyframe_insert(k, frame=f["frame"])

plates = sorted(Path(cam["plate_dir"]).glob("*" + Path(cam["plate_pattern"]).suffix))
img = None
if plates:
    img = bpy.data.images.load(str(plates[0])); img.source = "SEQUENCE"
    cd.show_background_images = True; bg = cd.background_images.new(); bg.image = img
    bg.image_user.frame_duration = cam["frame_end"]; bg.image_user.frame_start = 1; bg.image_user.use_auto_refresh = True


def projected(ob):
    if "Projected" not in ob.data.uv_layers:
        ob.data.uv_layers.new(name="Projected")
    mod = ob.modifiers.new("PlateProjection", "UV_PROJECT"); mod.uv_layer = "Projected"
    mod.aspect_x, mod.aspect_y = cam["width"], cam["height"]; mod.projector_count = 1; mod.projectors[0].object = co
    m = bpy.data.materials.new("Plate_" + ob.name); m.use_nodes = True; nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    em = nt.nodes.new("ShaderNodeEmission"); em.location = (350, 0)
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.location = (0, 0)
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "Projected"; uv.location = (-250, 0)
    if img:
        tx.image = img; tx.image_user.frame_duration = cam["frame_end"]; tx.image_user.frame_start = 1; tx.image_user.use_auto_refresh = True
    nt.links.new(uv.outputs["UV"], tx.inputs["Vector"]); nt.links.new(tx.outputs["Color"], em.inputs["Color"])
    emit_out = em.outputs["Emission"]
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (800, 0)
    lp = nt.nodes.new("ShaderNodeLightPath"); lp.location = (300, 250)
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse"); diff.location = (350, -150)
    diff.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    mix = nt.nodes.new("ShaderNodeMixShader"); mix.location = (600, 0)
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(emit_out, mix.inputs[1])            # what the character sees: the plate
    nt.links.new(diff.outputs["BSDF"], mix.inputs[2])  # what the shadow catcher measures
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    ob.data.materials.append(m); ob.is_shadow_catcher = True


if (here / "planes.obj").exists():
    bpy.ops.wm.obj_import(filepath=str(here / "planes.obj"), forward_axis="Y", up_axis="Z")
    for ob in bpy.context.selected_objects:
        ob.parent = root; projected(ob)
if (here / "proxy.ply").exists():
    bpy.ops.wm.ply_import(filepath=str(here / "proxy.ply"))
    ob = bpy.context.view_layer.objects.active; ob.name = "EnvironmentProxy"; ob.parent = root; projected(ob)
if (here / "points.ply").exists():
    bpy.ops.wm.ply_import(filepath=str(here / "points.ply"))
    ob = bpy.context.view_layer.objects.active; ob.name = "TrackPoints"; ob.parent = root; ob.hide_render = True
if (here / "env.hdr").exists():
    w = bpy.data.worlds.new("ShotWorld"); scene.world = w; w.use_nodes = True
    tex = w.node_tree.nodes.new("ShaderNodeTexEnvironment"); tex.image = bpy.data.images.load(str(here / "env.hdr"))
    w.node_tree.links.new(tex.outputs["Color"], w.node_tree.nodes["Background"].inputs["Color"])
if cam.get("character_position"):
    a = bpy.data.objects.new("CharacterAnchor", None); a.empty_display_type = "ARROWS"; a.location = cam["character_position"]; a.parent = root; scene.collection.objects.link(a)


# ---- compositor: plate x shadow catcher, character over the top -------------------------
def build_comp():
    if bpy.app.version >= (5, 0, 0):
        nt = bpy.data.node_groups.new("ShotComp", "CompositorNodeTree")
        scene.compositing_node_group = nt
        nt.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out = nt.nodes.new("NodeGroupOutput")
        try:
            scene.render.use_compositing = True
        except AttributeError:
            pass
    else:
        scene.use_nodes = True
        nt = scene.node_tree
        for nd in list(nt.nodes):
            nt.nodes.remove(nd)
        out = nt.nodes.new("CompositorNodeComposite")
    rl = nt.nodes.new("CompositorNodeRLayers"); rl.location = (-600, 200)
    plate = nt.nodes.new("CompositorNodeImage"); plate.location = (-600, -200); plate.label = "Plate"
    plate.image = img; plate.frame_duration = cam["frame_end"]; plate.frame_start = 1; plate.use_auto_refresh = True
    # multiply node: CompositorNodeMixRGB in 4.x, ShaderNodeMix in 5.x
    try:
        mul = nt.nodes.new("CompositorNodeMixRGB"); mul.blend_type = "MULTIPLY"
        a, b, res = mul.inputs[1], mul.inputs[2], mul.outputs[0]
        mul.inputs[0].default_value = 1.0
    except RuntimeError:
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
        ins = [s_ for s_ in mul.inputs if s_.enabled]; outs = [s_ for s_ in mul.outputs if s_.enabled]
        ins[0].default_value = 1.0
        a, b, res = ins[1], ins[2], outs[0]
    mul.location = (-250, -100); mul.label = "Plate x shadows"
    over = nt.nodes.new("CompositorNodeAlphaOver"); over.location = (50, 0)
    viewer = nt.nodes.new("CompositorNodeViewer"); viewer.location = (350, -200)
    out.location = (350, 0)
    nt.links.new(plate.outputs["Image"], a)
    sc = rl.outputs.get("Shadow Catcher")
    if sc is not None:
        nt.links.new(sc, b)
    else:
        b.default_value = (1, 1, 1, 1)
        print("vfxpipe: no Shadow Catcher pass socket — shadows not multiplied in")
    over_in = [s_ for s_ in over.inputs if s_.enabled and s_.type == "RGBA"]
    nt.links.new(res, over_in[0])
    nt.links.new(rl.outputs["Image"], over_in[1])
    nt.links.new(over.outputs[0], out.inputs[0])
    nt.links.new(over.outputs[0], viewer.inputs[0])


if img:
    try:
        build_comp()
        print("vfxpipe: compositor set up (plate x shadow catcher, character over)")
    except Exception as e:  # keep the rest of the import even if the node API differs
        print("vfxpipe: compositor setup failed, build it by hand:", e)

print("vfxpipe bundle imported")
