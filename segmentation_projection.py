import bpy

extrude_dist = 10
f = 1151.0 # focal length in pixels
z_val = 0.1 # distance from pinhole to start extrude (meters)
svg2pix = 2835.4 # Converts svg units to pixels (pixels/svg_units)
# ^calculated using 448 px = 0.158 m

##################################################################
# Mask 1
##################################################################
bpy.ops.import_curve.svg(filepath='/home/nlbucki/Documents/CS280/3D_Reconstruction.git/bootMask1.svg')

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
bpy.context.object.active_material.diffuse_color = (0.4, 0.4, 0.4)

# Extrude
bpy.ops.object.mode_set(mode   = 'EDIT')
bpy.ops.mesh.select_mode(type  = 'FACE')
bpy.ops.mesh.select_all(action = 'SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, extrude_dist-z_val)})

# Scale
bpy.ops.transform.resize(value=((extrude_dist-z_val)/z_val, (extrude_dist-z_val)/z_val, 1))

# Simplify mesh
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.dissolve_limited()

curve.select = False
# bpy.ops.object.mode_set(mode = 'OBJECT')

##################################################################
# Mask 2
##################################################################
bpy.ops.import_curve.svg(filepath='/home/nlbucki/Documents/CS280/3D_Reconstruction.git/bootMask2.svg')

# Select curve
curve2 = bpy.data.objects['Curve.001']
bpy.context.scene.objects.active = curve2
curve2.select = True
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
bpy.context.object.active_material.diffuse_color = (0.3, 0.3, 0.3)

# Extrude
bpy.ops.object.mode_set(mode   = 'EDIT')
bpy.ops.mesh.select_mode(type  = 'FACE')
bpy.ops.mesh.select_all(action = 'SELECT')
bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, extrude_dist-z_val)})

# Scale
bpy.ops.transform.resize(value=((extrude_dist-z_val)/z_val, (extrude_dist-z_val)/z_val, 1))

# Simplify mesh
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.dissolve_limited()

# Rotate/Translate
bpy.ops.object.mode_set(mode = 'OBJECT')
bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
bpy.ops.transform.translate(value=(-0.19878306, 0.83551793, 0.51224513))
bpy.ops.transform.rotate(value=1.2678681433439856, axis=(0.976963675128, 0.180246910261, 0.114249852615))

asdf

bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["Curve"]
bpy.context.object.modifiers["Boolean"].solver = 'CARVE'
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Boolean")


curve2.select = False
curve.select = True
bpy.ops.object.delete(use_global=False)

for ob in bpy.data.objects:
    print (ob.name)