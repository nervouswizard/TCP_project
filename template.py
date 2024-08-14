import bpy
import os
import math

base_path = os.path.join('squirrel_new') ### image name
layer_dir = os.path.join(base_path + 'layer')
alpha_dir = os.path.join(base_path + 'alpha')
background_dir = os.path.join(base_path + 'background')
# project_name = 'human'

layer_num = 10  ### mask number
depth_paths = [[] for _ in range(layer_num)]
rgb_paths = [[] for _ in range(layer_num)]
background_paths = []
H = 0
W = 0
scale = 10  ### resize scale
layer_offset = 0.25 ###
displace_strength = 11 ###

# print(depth_paths)

# region Read files
for file in sorted(os.listdir(layer_dir)):
    fullpath = os.path.join(layer_dir, file)
    prefix, suffix = int(file.split('_')[1].split('-')[0])-1, int(file.split('-')[1].split('.')[0])-1
    # print(file, prefix, suffix)
    rgb_paths[prefix].append(fullpath)
        
for file in sorted(os.listdir(alpha_dir)):
    fullpath = os.path.join(alpha_dir, file)
    prefix, suffix = int(file.split('_')[1].split('-')[0])-1, int(file.split('-')[1].split('.')[0])-1
    # print(file, prefix, suffix)
    depth_paths[prefix].append(fullpath)

for file in sorted(os.listdir(background_dir)):
    fullpath = os.path.join(background_dir, file)
    background_paths.append(fullpath)

# print(depth_paths)
# print(rgb_paths)
# endregion

# region Initialize blender scene
# Create a new Blender scene
scene = bpy.context.scene

# Set the units to meters
bpy.context.scene.unit_settings.system = 'METRIC'

# endregion

# region Clean environment
# # Select all objects
# bpy.ops.object.select_all(action='SELECT')
# # Delete all objects
# bpy.ops.object.delete()

# Delete all objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj)
# Delete all materials
for material in bpy.data.materials:
    bpy.data.materials.remove(material)
# Delete all textures
for texture in bpy.data.textures:
    bpy.data.textures.remove(texture)
# endregion

# region Read background exr
# Load the EXR file
exr_image = bpy.data.images.load(background_paths[0])
# Create a new node tree for the world
world_node_tree = bpy.data.worlds['World'].node_tree
# Clear the node tree
world_node_tree.nodes.clear()
# Create a new environment texture node
env_texture_node = world_node_tree.nodes.new(type="ShaderNodeTexEnvironment")
env_texture_node.image = exr_image
env_texture_node.location = (0, 0)
# Create a new background node
background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
background_node.location = (400, 0)
# Create a new output node
output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")
output_node.location = (800, 0)
# Link the nodes
world_node_tree.links.new(env_texture_node.outputs[0], background_node.inputs[0])
world_node_tree.links.new(background_node.outputs[0], output_node.inputs[0])
# endregion

# bpy.ops.object.empty_add(type='CUBE', location=(0, 0, 0), scale=(50, 50, 50))
# empty_cube = bpy.context.active_object

# region Create plane meshes
for i in range(len(rgb_paths)):
    for j in range(len(rgb_paths[i])):
        # Create a new plane mesh
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, -i*layer_offset, 0), scale=(1, 1, 1))
        plane = bpy.context.active_object
        plane.name = f"Plane {i}-{j}"
        plane.rotation_euler = (math.pi / 2, 0, 0)  # Add this line     

        # Subdivide the plane mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=10000)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Add a displace modifier with an image texture
        displacement_modifier = plane.modifiers.new("Displace", 'DISPLACE')
        displacement_texture = bpy.data.textures.new(f"DisplaceTexture {i}-{j}", type="IMAGE")
        image = bpy.data.images.load(depth_paths[i][j])
        # Get the dimensions of the image
        W, H = image.size
        displacement_texture.image = image
        displacement_modifier.texture = displacement_texture
        # Set the strength of the displace texture to 0.5
        displacement_modifier.strength = displace_strength
        
        # Add a Simple Deform modifier to bend the mesh
        # simple_deform_modifier = plane.modifiers.new("Simple Deform", 'SIMPLE_DEFORM')
        # simple_deform_modifier.angle = math.pi / 2 # 45 degrees
        # simple_deform_modifier.deform_axis = 'Z'
        # simple_deform_modifier.origin = empty_cube
        # simple_deform_modifier.deform_method = 'BEND'

        # Set the plane's dimensions to match the image size
        plane.dimensions = (W/scale, H/scale, 0)
        
        # Set the material
        mat = bpy.data.materials.new(f"Material {i}-{j}")
        plane.data.materials.append(mat)

        # Set the Blender mode to Alpha Blend
        mat.blend_method = 'BLEND'
        
        
        mat.use_nodes = True
        # Create a new shader node tree
        node_tree = mat.node_tree
        node_tree.nodes.clear()

        # Create a new image texture node
        image_texture_node = node_tree.nodes.new(type="ShaderNodeTexImage")
        image = bpy.data.images.load(rgb_paths[i][j])
        image_texture_node.image = image
        image_texture_node.location = (-400, 0)

        # Create a new principled BSDF node
        principled_bsdf_node = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        principled_bsdf_node.location = (0, 0)

        # Create a new output node
        output_node = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (400, 0)

        # Link the nodes
        node_tree.links.new(image_texture_node.outputs[0], principled_bsdf_node.inputs[0])
        node_tree.links.new(image_texture_node.outputs[1], principled_bsdf_node.inputs[21])
        node_tree.links.new(principled_bsdf_node.outputs[0], output_node.inputs[0])
# endregion

# # region Camera
# # Create a BezierCircle
# curve = bpy.data.curves.new('BezierCircle', type='CURVE')
# curve.dimensions = '3D'
# curve_object = bpy.data.objects.new('BezierCircle', curve)
# bpy.context.collection.objects.link(curve_object)


# # Create a BezierCircle
# bezier_circle = curve.splines.new(type='BEZIER')
# bezier_circle.bezier_points.add(3)
# bezier_circle.bezier_points[0].co = (0, 0, 0)
# bezier_circle.bezier_points[1].co = (1, 0, 0)
# bezier_circle.bezier_points[2].co = (1, 1, 0)
# bezier_circle.bezier_points[3].co = (0, 1, 0)

# # Create a new camera
# camera = bpy.data.cameras.new("Camera")
# camera_object = bpy.data.objects.new("Camera", camera)
# bpy.context.collection.objects.link(camera_object)
# camera_object.location = bezier_circle.bezier_points[0].co

# # Create a follow path constraint for the camera
# constraint = camera_object.constraints.new('FOLLOW_PATH')
# constraint.target = curve_object
# constraint.use_fixed_location = True

# # Set the camera's direction to track the meshes
# track_constraint = camera_object.constraints.new('TRACK_TO')
# track_constraint.target = plane
# track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
# track_constraint.up_axis = 'UP_Y'

# # # Animate the camera's movement along the curve
# # bpy.context.scene.frame_end = 100
# # curve_object.data.path_duration = 100
# # camera_object.location = curve_object.location
# # camera_object.rotation_euler = (0, 0, 0)
# # endregion

# Set the active object to the first plane
# bpy.context.view_layer.objects.active = planes[0]

# Save the Blender file
# bpy.context.scene.name = project_name
# bpy.ops.wm.save_as_mainfile(project_name + ".blend")