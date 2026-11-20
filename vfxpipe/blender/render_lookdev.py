"""blender -b --python render_lookdev.py -- <env.hdr> <out.png> [GPU|CPU]

A grey ball, a chrome ball and a shadow-catching floor under the environment
map — the same test a VFX supervisor shoots on set, so you can judge colour,
direction and softness of the light before the character exists.
"""
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
env, out = Path(args[0]), Path(args[1])
device = args[2] if len(args) > 2 else "CPU"

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = 960, 540
scene.render.film_transparent = False

world = bpy.data.worlds.new("W"); scene.world = world; world.use_nodes = True
tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
tex.image = bpy.data.images.load(str(env))
world.node_tree.links.new(tex.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])


def ball(name, x, rough, metal):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(x, 0, 0.5), segments=64, ring_count=32)
    ob = bpy.context.object; ob.name = name
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1) if not metal else (0.95, 0.95, 0.95, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    ob.data.materials.append(m)


ball("GreyBall", -0.7, 0.6, 0.0)
ball("ChromeBall", 0.7, 0.05, 1.0)
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
floor = bpy.context.object; floor.name = "Floor"; floor.is_shadow_catcher = True

cam_data = bpy.data.cameras.new("C"); cam_data.lens = 50
cam = bpy.data.objects.new("C", cam_data); scene.collection.objects.link(cam); scene.camera = cam
cam.location = (0, -4.2, 1.4)
cam.rotation_euler = (1.36, 0, 0)

if device == "GPU":
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"; (prefs.refresh_devices if hasattr(prefs, "refresh_devices") else prefs.get_devices)()
        for d in prefs.devices:
            d.use = d.type == "CUDA"
        scene.cycles.device = "GPU"
    except Exception:
        pass

scene.render.filepath = str(out)
bpy.ops.render.render(write_still=True)
