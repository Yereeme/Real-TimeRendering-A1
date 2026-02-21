import bpy
import math

# --- params you tweak for the CPU bottleneck ---
GRID_X = 120          # 120 * 120 = 14,400 objects (good CPU stress)
GRID_Y = 120
SPACING = 1.2         # keep objects spaced to avoid overlap
CUBE_SIZE = 0.35      # small geometry (keep GPU light)

OUT_BLEND = r"C:\Users\aykone\A1\code\scenes\cpu_test.blend"

# --- reset scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- optional: clean default collections/objects ---
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# --- make a camera (nice stable view) ---
bpy.ops.object.camera_add(location=(GRID_X * SPACING * 0.5, -GRID_Y * SPACING * 1.2, GRID_Y * SPACING * 0.6))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(65), 0, math.radians(0))
bpy.context.scene.camera = cam

# --- simple light ---
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0

# --- create many separate cube OBJECTS (this is the CPU bottleneck lever) ---
# Use linked data so mesh memory isn't huge; still many objects / draw calls.
bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE, location=(0, 0, 0))
proto = bpy.context.active_object
proto.name = "CubeProto"
proto.data.name = "CubeMesh"

# Put proto out of the way (or delete later)
proto.location = (-9999, -9999, -9999)

for y in range(GRID_Y):
    for x in range(GRID_X):
        obj = proto.copy()
        obj.data = proto.data  # share the same mesh datablock (important)
        obj.location = (x * SPACING, y * SPACING, 0.0)
        bpy.context.scene.collection.objects.link(obj)

# --- save blend ---
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
print(f"Saved CPU test scene to: {OUT_BLEND}")
print(f"Object count (incl proto/cam/light): {len(bpy.data.objects)}")