import bpy

extrude_dist = 10 # meters
f = 1151.0 # focal length in pixels
z_val = 1 # distance from pinhole to start extrude (meters)
svg2pix = 2835.4 # Converts svg units to pixels (pixels/svg_units)
# ^calculated using 448 px = 0.158 m

##################################################################
# Cup 1
##################################################################
bpy.ops.import_curve.svg(filepath='/home/nlbucki/Downloads/Cup.svg')

# Select curve
curve = bpy.data.objects['Curve']
bpy.context.scene.objects.active = curve
curve.select = True
# Translate 448 px x 448 px SVG to put origin at center and put image 0.1 m in front of pinhole
bpy.ops.transform.translate(value=(-0.158/2.0, 0.158/2.0, z_val))
# Move origin to center
bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
# Resize image to correct size given z_val
bpy.ops.transform.resize(value=(svg2pix*z_val/f, svg2pix*z_val/f, 1))

# Convert curve to mesh
bpy.ops.object.convert(target='MESH')

# Change color
bpy.context.object.active_material.diffuse_color = (0.464668, 0.464668, 0.464668)

# Extrude
bpy.ops.object.mode_set(mode   = 'EDIT')
bpy.ops.mesh.select_mode(type  = 'FACE')
bpy.ops.mesh.select_all(action = 'SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, extrude_dist-z_val)})

# Scale
bpy.ops.transform.resize(value=(extrude_dist, extrude_dist, 1))

# Simplify mesh
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.dissolve_limited()

curve.select = False
bpy.ops.object.mode_set(mode = 'OBJECT')
asdf

##################################################################
# Cup 2
##################################################################

bpy.ops.import_curve.svg(filepath='/home/nlbucki/Downloads/Cup2.svg')

# Select curve
curve2 = bpy.data.objects['Curve.001']
bpy.context.scene.objects.active = curve2
curve2.select = True
bpy.ops.transform.translate(value=(-0.158/2, 0.158/2, 0))

# Convert curve to mesh
bpy.ops.object.convert(target='MESH')

# Change color
bpy.context.object.active_material.diffuse_color = (0.464668, 0.464668, 0.464668)

# Extrude
bpy.ops.object.mode_set(mode   = 'EDIT')
bpy.ops.mesh.select_mode(type  = 'FACE')
bpy.ops.mesh.select_all(action = 'SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, extrude_val)})

# Scale
bpy.ops.transform.resize(value=(2.29934, 2.29934, 2.29934))

# Simplify mesh
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.dissolve_limited()

# Rotate/Translate
bpy.ops.object.mode_set(mode = 'OBJECT')
bpy.ops.transform.translate(value=(0.20511064, 0.0926139, 0.97434711))
bpy.ops.transform.rotate(value=0.4172196360817166, axis=(0.200630991023, 0.0602409660758, 0.977812983882))

#bpy.ops.object.modifier_add(type='BOOLEAN')
#bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Curve"]
#bpy.context.object.modifiers["Boolean"].solver = 'CARVE'
#bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")


#curve2.select = False
#curve.select = True
#bpy.ops.object.delete(use_global=False)

for ob in bpy.data.objects:
    print (ob.name)