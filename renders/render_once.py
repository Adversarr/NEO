"""Usage:
python -u scripts/eig/demo_plot_ambiguity.py --out_dir tmp/ambiguity_mesh_pretty_test --sphere_subdiv 2 --sphere_eig_k 8 --mesh_target_faces 2000 --mesh_eig_k 15 --mesh_eig_index 6 --render_resolution 1000 1000 --render_samples 128 --render_cam_pos 0 1 2 --render_world_up 0 1 0 --render_area_light_pos -1 4 8 --render_area_light_energy=300
"""

import bpy
import sys
import argparse
import math
from pathlib import Path

def clean_scene():
    """Clear all objects in the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera(location, rotation_euler, fov=50, is_ortho=False, ortho_scale=6.0):
    """Set up the camera"""
    bpy.ops.object.camera_add(location=location, rotation=rotation_euler)
    cam = bpy.context.object
    if is_ortho:
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = float(ortho_scale)
    else:
        cam.data.type = 'PERSP'
        cam.data.lens_unit = 'FOV'
        cam.data.angle = math.radians(fov)
    bpy.context.scene.camera = cam
    return cam

def setup_light(
    location,
    energy=1000,
    type="POINT",
    rotation_euler=None,
    color=(1.0, 1.0, 1.0),
    size=1.0,
    size_y=None,
    shadow_soft_size=0.0,
    spread=1.0,
    angle=0.0,
):
    """Set up the light sources"""
    if rotation_euler is None:
        bpy.ops.object.light_add(type=type, location=location)
    else:
        bpy.ops.object.light_add(type=type, location=location, rotation=rotation_euler)

    light = bpy.context.object
    light.data.energy = energy

    if hasattr(light.data, "color"):
        light.data.color = color

    if type == "AREA":
        if hasattr(light.data, "shape"):
            light.data.shape = "RECTANGLE" if size_y is not None else "SQUARE"
        if hasattr(light.data, "size"):
            light.data.size = size
        if size_y is not None and hasattr(light.data, "size_y"):
            light.data.size_y = size_y
        if hasattr(light.data, "spread"):
            light.data.spread = float(spread)

    if type == "POINT":
        if hasattr(light.data, "shadow_soft_size"):
            light.data.shadow_soft_size = float(shadow_soft_size)

    if type == "SUN":
        if hasattr(light.data, "angle"):
            light.data.angle = float(angle)

    return light


def setup_world(background_rgb=None, world_strength=0.0, transparent=True):
    scene = bpy.context.scene
    scene.render.film_transparent = transparent

    if scene.world is None:
        scene.world = bpy.data.worlds.new(name="World")

    world = scene.world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background_node = nodes.get("Background")
    if background_node is None:
        background_node = nodes.new(type="ShaderNodeBackground")

    output_node = nodes.get("World Output")
    if output_node is None:
        output_node = nodes.new(type="ShaderNodeOutputWorld")

    try:
        surface_in = output_node.inputs["Surface"]
    except Exception:
        surface_in = None

    try:
        background_out = background_node.outputs["Background"]
    except Exception:
        background_out = None

    if surface_in is not None and background_out is not None:
        has_surface_link = any(link.to_node == output_node and link.to_socket == surface_in for link in links)
        if not has_surface_link:
            links.new(background_out, surface_in)

    if background_rgb is None:
        background_rgb = (0.0, 0.0, 0.0)

    if "Color" in background_node.inputs:
        background_node.inputs["Color"].default_value = (*background_rgb, 1.0)
    if "Strength" in background_node.inputs:
        background_node.inputs["Strength"].default_value = world_strength


def _degrees_to_radians(vec3_deg):
    return [math.radians(float(v)) for v in vec3_deg]


def _look_at_euler(location, target):
    from mathutils import Vector

    direction = Vector(target) - Vector(location)
    if direction.length == 0:
        return (0.0, 0.0, 0.0)

    rot_quat = direction.to_track_quat("-Z", "Y")
    return rot_quat.to_euler()


def _get_object_center_and_radius(obj):
    from mathutils import Vector

    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector(
        (
            min(v.x for v in world_corners),
            min(v.y for v in world_corners),
            min(v.z for v in world_corners),
        )
    )
    max_corner = Vector(
        (
            max(v.x for v in world_corners),
            max(v.y for v in world_corners),
            max(v.z for v in world_corners),
        )
    )
    center = (min_corner + max_corner) * 0.5
    radius = (max_corner - min_corner).length * 0.5
    return (center.x, center.y, center.z), float(radius)


def _get_object_bounds(obj):
    from mathutils import Vector

    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector(
        (
            min(v.x for v in world_corners),
            min(v.y for v in world_corners),
            min(v.z for v in world_corners),
        )
    )
    max_corner = Vector(
        (
            max(v.x for v in world_corners),
            max(v.y for v in world_corners),
            max(v.z for v in world_corners),
        )
    )
    return (float(min_corner.x), float(min_corner.y), float(min_corner.z)), (
        float(max_corner.x),
        float(max_corner.y),
        float(max_corner.z),
    )


def _normalize_vec3(vec3):
    from mathutils import Vector

    v = Vector(vec3)
    if v.length == 0:
        return (0.0, -1.0, 0.0)
    v.normalize()
    return (float(v.x), float(v.y), float(v.z))


def _auto_frame_camera(center, radius, fov_deg, cam_dir, margin=1.2, distance_scale=1.0):
    from mathutils import Vector

    radius = max(float(radius), 1e-6)
    half_angle = math.radians(float(fov_deg)) * 0.5
    dist = (radius / max(math.tan(half_angle), 1e-6)) * float(margin) * float(distance_scale)

    cam_dir_n = Vector(_normalize_vec3(cam_dir))
    cam_pos = Vector(center) - cam_dir_n * dist
    cam_rot = _look_at_euler(cam_pos, center)

    clip_start = max(radius * 0.001, 1e-4)
    clip_end = max(radius * 100.0, dist * 10.0)
    return (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z)), cam_rot, clip_start, clip_end


def _add_ground_plane(
    center,
    radius,
    obj_min,
    ground_normal,
    ground_point=None,
    offset=0.05,
    size=0.0,
    shadow_catcher=True,
):
    from mathutils import Vector

    n = Vector(_normalize_vec3(ground_normal))
    r = max(float(radius), 1e-6)
    offset_abs = float(offset)

    if ground_point is None:
        if abs(n.z) > 0.9:
            ground_point = (float(center[0]), float(center[1]), float(obj_min[2]) - offset_abs)
        else:
            ground_point = (Vector(center) - n * (r + offset_abs))
            ground_point = (float(ground_point.x), float(ground_point.y), float(ground_point.z))

    plane_size = float(size)
    if plane_size <= 0:
        plane_size = 6.0 * r

    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=ground_point)
    plane_obj = bpy.context.object

    z_axis = Vector((0.0, 0.0, 1.0))
    rot = z_axis.rotation_difference(n)
    plane_obj.rotation_euler = rot.to_euler()

    if shadow_catcher:
        if hasattr(plane_obj, "is_shadow_catcher"):
            plane_obj.is_shadow_catcher = True
        if hasattr(plane_obj, "cycles") and hasattr(plane_obj.cycles, "is_shadow_catcher"):
            plane_obj.cycles.is_shadow_catcher = True

    return plane_obj
def create_geometry_nodes(
    obj,
    point_radius,
    icosphere_subdivisions=1,
    original_col_name="Col",
    color_override_rgba=None,
):
    """
    Programmatically build geometry node tree (fix/adapt for Blender 4.0+)
    """
    # 1. Add geometry node modifier
    modifier = obj.modifiers.new(name="PointCloudNodes", type='NODES')
    node_group = bpy.data.node_groups.new("PC_NodeGroup", 'GeometryNodeTree')
    modifier.node_group = node_group

    # --- Key fix: explicitly define interface (required for Blender 4.0+) ---
    # In 4.0+, newly created Groups have no inputs/outputs and must be added manually
    # Add input socket (Input Socket) -> corresponds to output of Group Input node
    if hasattr(node_group, "interface"): # Blender 4.0+ API
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        # If you want the radius to be adjustable in the modifier panel, add this line:
        # node_group.interface.new_socket(name="Point Radius", in_out='INPUT', socket_type='NodeSocketFloat')
    else: # Legacy compatibility (3.x)
        node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        node_group.outputs.new('NodeSocketGeometry', 'Geometry')

    # Get node tree reference
    nodes = node_group.nodes
    links = node_group.links

    # Clear default nodes (even though newly created ones are empty, just to be safe)
    for node in nodes:
        nodes.remove(node)

    # --- Create nodes ---

    # Group Input (now has 'Geometry' output port)
    node_in = nodes.new('NodeGroupInput')
    node_in.location = (-600, 0)

    # Group Output (now has 'Geometry' input port)
    node_out = nodes.new('NodeGroupOutput')
    node_out.location = (400, 0)

    # Mesh to Points (clean data)
    node_mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
    node_mesh_to_points.mode = 'VERTICES'
    node_mesh_to_points.location = (-400, 0)

    # Input Named Attribute (read original color Col)
    node_input_attr = nodes.new('GeometryNodeInputNamedAttribute')
    node_input_attr.data_type = 'FLOAT_COLOR'
    node_input_attr.inputs['Name'].default_value = original_col_name
    node_input_attr.location = (-400, -200)

    # Store Named Attribute (key step: store as instance attribute)
    node_store_attr = nodes.new('GeometryNodeStoreNamedAttribute')
    node_store_attr.data_type = 'FLOAT_COLOR'
    node_store_attr.domain = 'POINT'
    node_store_attr.inputs['Name'].default_value = "output_color"
    node_store_attr.location = (-200, 0)

    # Ico Sphere (used as instance sphere)
    node_sphere = nodes.new('GeometryNodeMeshIcoSphere')
    node_sphere.inputs['Radius'].default_value = point_radius
    node_sphere.inputs['Subdivisions'].default_value = int(icosphere_subdivisions)
    node_sphere.location = (-200, -300)

    node_set_shade_smooth = None
    try:
        node_set_shade_smooth = nodes.new('GeometryNodeSetShadeSmooth')
        node_set_shade_smooth.location = (-40, -300)
        for key in ("Shade Smooth", "Smooth"):
            if key in node_set_shade_smooth.inputs:
                try:
                    node_set_shade_smooth.inputs[key].default_value = True
                except Exception:
                    pass
        if "Selection" in node_set_shade_smooth.inputs:
            try:
                node_set_shade_smooth.inputs["Selection"].default_value = True
            except Exception:
                pass
    except Exception:
        node_set_shade_smooth = None
    
    # Instance on Points
    node_instance = nodes.new('GeometryNodeInstanceOnPoints')
    node_instance.location = (0, 0)
    
    # Set Material
    node_set_mat = nodes.new('GeometryNodeSetMaterial')
    node_set_mat.location = (200, 0)


    # --- Connect nodes ---
    # Note: we use inputs[0] / outputs[0] index connections here, safer than using names,
    # because Blender sometimes auto-appends suffixes to names (e.g. Geometry.001)

    # Input(Geometry) -> Mesh to Points(Mesh)
    # node_in.outputs[0] corresponds to the first INPUT we defined in the interface
    links.new(node_in.outputs[0], node_mesh_to_points.inputs['Mesh'])
    
    # Mesh to Points -> Store Attribute (Geometry)
    links.new(node_mesh_to_points.outputs['Points'], node_store_attr.inputs['Geometry'])
    if color_override_rgba is not None:
        node_override_color = None
        # Try Combine Color first (supports RGBA)
        try:
            node_override_color = nodes.new('FunctionNodeCombineColor')
            node_override_color.inputs['Red'].default_value = color_override_rgba[0]
            node_override_color.inputs['Green'].default_value = color_override_rgba[1]
            node_override_color.inputs['Blue'].default_value = color_override_rgba[2]
            node_override_color.inputs['Alpha'].default_value = color_override_rgba[3]
        except Exception:
            # Fallback to Input Color (RGB only)
            try:
                node_override_color = nodes.new('FunctionNodeInputColor')
                if hasattr(node_override_color, "color"):
                    node_override_color.color = color_override_rgba[:3]
            except Exception:
                node_override_color = None

        if node_override_color is not None:
            node_override_color.location = (-400, -360)
            links.new(node_override_color.outputs[0], node_store_attr.inputs['Value'])
        else:
            try:
                node_store_attr.inputs['Value'].default_value = tuple(color_override_rgba)
            except Exception:
                pass
    else:
        node_switch = nodes.new('GeometryNodeSwitch')
        for input_type in ("RGBA", "COLOR", "VECTOR"):
            try:
                node_switch.input_type = input_type
                break
            except Exception:
                continue
        node_switch.location = (-260, -200)

        exists_out = node_input_attr.outputs.get("Exists")
        if exists_out is None and len(node_input_attr.outputs) > 1:
            exists_out = node_input_attr.outputs[1]

        attr_out = node_input_attr.outputs.get("Attribute")
        if attr_out is None and len(node_input_attr.outputs) > 0:
            attr_out = node_input_attr.outputs[0]

        if exists_out is not None:
            links.new(exists_out, node_switch.inputs[0])

        try:
            node_switch.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)
        except Exception:
            pass

        if attr_out is not None:
            links.new(attr_out, node_switch.inputs[2])

        links.new(node_switch.outputs[0], node_store_attr.inputs['Value'])
    
    # Store Attribute -> Instance on Points (Points)
    links.new(node_store_attr.outputs['Geometry'], node_instance.inputs['Points'])
    
    # Sphere -> Instance on Points (Instance)
    if node_set_shade_smooth is not None:
        try:
            sphere_out = node_sphere.outputs.get('Mesh', None)
            if sphere_out is None and len(node_sphere.outputs) > 0:
                sphere_out = node_sphere.outputs[0]
            smooth_in = node_set_shade_smooth.inputs.get('Geometry', None)
            if smooth_in is None and len(node_set_shade_smooth.inputs) > 0:
                smooth_in = node_set_shade_smooth.inputs[0]
            smooth_out = node_set_shade_smooth.outputs.get('Geometry', None)
            if smooth_out is None and len(node_set_shade_smooth.outputs) > 0:
                smooth_out = node_set_shade_smooth.outputs[0]
            if sphere_out is not None and smooth_in is not None and smooth_out is not None:
                links.new(sphere_out, smooth_in)
                links.new(smooth_out, node_instance.inputs['Instance'])
            else:
                links.new(node_sphere.outputs['Mesh'], node_instance.inputs['Instance'])
        except Exception:
            links.new(node_sphere.outputs['Mesh'], node_instance.inputs['Instance'])
    else:
        links.new(node_sphere.outputs['Mesh'], node_instance.inputs['Instance'])
    
    # Instance on Points -> Set Material
    links.new(node_instance.outputs['Instances'], node_set_mat.inputs['Geometry'])
    
    # Set Material -> Output(Geometry)
    # node_out.inputs[0] corresponds to the first OUTPUT we defined in the interface
    links.new(node_set_mat.outputs['Geometry'], node_out.inputs[0])
    
    return node_set_mat

def create_shader_material(
    mat_name="PC_Material",
    roughness=0.5,
    specular=0.3,
    metallic=0.0,
    emission_strength=0.0,
    subsurface_weight=0.05,
    use_ao=False,
    attribute_type='INSTANCER',
    attribute_name="output_color",
):
    """Create shader material, read output_color attribute"""
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get Principled BSDF and Output nodes
    bsdf = nodes.get('Principled BSDF')
    output_node = nodes.get("Material Output")
    if output_node is None:
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        try:
            surface_in = output_node.inputs["Surface"]
        except Exception:
            surface_in = None
        try:
            bsdf_out = bsdf.outputs["BSDF"]
        except Exception:
            bsdf_out = None
        if surface_in is not None and bsdf_out is not None:
            links.new(bsdf_out, surface_in)

    # Create Attribute node
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_type = attribute_type # Key: Instancer type or Geometry
    attr_node.attribute_name = attribute_name # Must match the name in the geometry nodes
    attr_node.location = (-500, 200)

    # Connect Attribute Color -> BSDF Base Color
    color_socket = attr_node.outputs['Color']
    
    if use_ao:
        ao_node = nodes.new('ShaderNodeAmbientOcclusion')
        ao_node.location = (-250, 200)
        # Use color as tint for AO
        links.new(color_socket, ao_node.inputs['Color'])
        color_socket = ao_node.outputs['Color']

    links.new(color_socket, bsdf.inputs['Base Color'])
    if "Alpha" in attr_node.outputs and "Alpha" in bsdf.inputs:
        links.new(attr_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = 'BLEND'
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = 'NONE'

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(specular)
    elif "IOR" in bsdf.inputs:
        v = float(specular)
        if v >= 1.0:
            ior_val = v
        else:
            f0 = max(0.0, min(0.08 * v, 0.999999))
            r = math.sqrt(f0)
            denom = max(1e-6, 1.0 - r)
            ior_val = (1.0 + r) / denom
        bsdf.inputs["IOR"].default_value = float(ior_val)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(metallic)

    if hasattr(mat, "cycles") and hasattr(mat.cycles, "subsurface_method"):
        for method in ("BURLEY", "CHRISTENSEN_BURLEY", "CHRISTENSEN-BURLEY"):
            try:
                mat.cycles.subsurface_method = method
                break
            except Exception:
                continue

    subsurface_socket_name = None
    for name in ("Subsurface Weight", "Subsurface"):
        if name in bsdf.inputs:
            subsurface_socket_name = name
            break
    if subsurface_socket_name is not None:
        try:
            bsdf.inputs[subsurface_socket_name].default_value = float(subsurface_weight)
        except Exception:
            pass

    if emission_strength and emission_strength > 0:
        if "Emission Color" in bsdf.inputs:
            links.new(attr_node.outputs["Color"], bsdf.inputs["Emission Color"])
        elif "Emission" in bsdf.inputs:
            links.new(attr_node.outputs["Color"], bsdf.inputs["Emission"])
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = float(emission_strength)

    return mat

def import_and_process_ply(
    filepath,
    radius,
    icosphere_subdivisions=1,
    original_col_name="Col",
    color_override_rgba=None,
    material_roughness=0.4,
    material_specular=0.5,
    material_metallic=0.0,
    material_emission_strength=0.0,
    use_ao=False,
):
    # Import PLY
    # Note: Blender 4.0+ uses wm.ply_import, older versions may use import_mesh.ply
    try:
        bpy.ops.wm.ply_import(filepath=filepath)
    except AttributeError:
        # Fallback for older Blender versions
        bpy.ops.import_mesh.ply(filepath=filepath)

    obj = bpy.context.selected_objects[0]

    # Detect whether it's a Mesh (has faces) or Point Cloud (no faces)
    is_mesh = False
    if hasattr(obj.data, "polygons") and len(obj.data.polygons) > 0:
        is_mesh = True
        print(f"Detected Mesh with {len(obj.data.polygons)} faces.")
    else:
        print("Detected Point Cloud (no faces).")

    # Determine material attribute source
    if is_mesh:
        mat_attr_type = 'GEOMETRY'
        mat_attr_name = original_col_name
    else:
        mat_attr_type = 'INSTANCER'
        mat_attr_name = "output_color"

    # Create material
    mat = create_shader_material(
        roughness=material_roughness,
        specular=material_specular,
        metallic=material_metallic,
        emission_strength=material_emission_strength,
        use_ao=use_ao,
        attribute_type=mat_attr_type,
        attribute_name=mat_attr_name,
    )

    if is_mesh:
        # If it's a Mesh, apply material directly
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Set smooth shading
        bpy.ops.object.shade_smooth()
    else:
        # If it's a point cloud, create geometry nodes and apply material
        set_mat_node = create_geometry_nodes(
            obj,
            radius,
            icosphere_subdivisions=icosphere_subdivisions,
            original_col_name=original_col_name,
            color_override_rgba=color_override_rgba,
        )
        set_mat_node.inputs['Material'].default_value = mat
    
    return obj

def main():
    # --- Process command-line arguments ---
    # Blender puts its own arguments in sys.argv, we need to filter out content before "--"
    if "--" not in sys.argv:
        args = []
    else:
        args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Render PLY Point Cloud with Blender")

    # Required arguments
    parser.add_argument('--input', type=str, nargs='+', required=True, help="Input .ply file path(s)")
    parser.add_argument('--output', type=str, required=True, help="Output image path (e.g. result.png)")

    # Optional arguments - scene
    parser.add_argument('--radius', type=float, default=0.01, help="Radius of the points (spheres)")
    parser.add_argument('--ico_subdiv', type=int, default=1, help="Icosphere subdivisions for each point")
    parser.add_argument('--col_name', type=str, default="Col", help="Name of color attribute in PLY")
    parser.add_argument('--swap_yz', action='store_true', help="Swap Y and Z coordinates after import")
    parser.add_argument('--obj_rot_deg', nargs=3, type=float, default=None, help="Custom object rotation in degrees (Euler X Y Z)")
    parser.add_argument('--obj_scale', nargs=3, type=float, default=None, help="Custom object scale (X Y Z)")
    parser.add_argument('--obj_location', nargs=3, type=float, default=None, help="Custom object location (X Y Z)")

    # Optional arguments - background/world
    parser.add_argument(
        '--background_rgb',
        nargs=3,
        type=float,
        default=None,
        help="Opaque background RGB (0..1). If omitted, render with transparent background.",
    )
    parser.add_argument(
        '--color-override',
        dest="color_override",
        nargs=4,
        type=float,
        default=None,
        help="Override point color as RGBA (0..1) when input has no color attribute",
    )
    parser.add_argument('--world_strength', type=float, default=0.0, help="World background strength")

    # Optional arguments - camera (X,Y,Z, RX,RY,RZ)
    parser.add_argument('--cam_pos', nargs=3, type=float, default=[0, -5, 1], help="Camera location X Y Z")
    parser.add_argument('--cam_rot', nargs=3, type=float, default=[math.radians(90), 0, 0], help="Camera rotation Euler X Y Z (in radians)")
    parser.add_argument('--cam_rot_deg', nargs=3, type=float, default=None, help="Camera rotation Euler X Y Z (in degrees)")
    parser.add_argument('--cam_target', nargs=3, type=float, default=None, help="Look-at target X Y Z")
    parser.add_argument('--look_at_center', action='store_true', help="Make camera look at point cloud center")
    parser.add_argument('--fov', type=float, default=50.0, help="Camera field of view (degrees)")
    parser.add_argument('--auto_frame', action='store_true', help="Auto frame camera and lights using bounding box")
    parser.add_argument('--frame_margin', type=float, default=1.2, help="Margin for auto framing")
    parser.add_argument('--auto_cam_dir', nargs=3, type=float, default=[0.0, -1.0, 0.25], help="Auto camera direction")
    parser.add_argument('--auto_distance_scale', type=float, default=1.0, help="Auto camera distance scale")
    parser.add_argument('--auto_lights', action='store_true', help="Auto place lights when auto framing")

    # Optional arguments - projection
    parser.add_argument('--orthographic', action='store_true', help="Use orthographic projection")
    parser.add_argument('--ortho_scale', type=float, default=6.0, help="Orthographic scale (if not auto-framed)")

    # Optional arguments - rendering
    parser.add_argument('--resolution', nargs=2, type=int, default=[1920, 1080], help="Resolution W H")
    parser.add_argument('--samples', type=int, default=128, help="Render samples")
    parser.add_argument('--engine', type=str, default='CYCLES', choices=['CYCLES', 'BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'], help="Render engine")
    parser.add_argument('--file_format', type=str, default="", choices=["", "PNG", "OPEN_EXR", "JPEG"], help="Output format override")
    parser.add_argument('--denoise', action='store_true', help="Enable Cycles denoising")
    parser.add_argument('--cycles_device', type=str, default='AUTO', choices=['AUTO', 'CPU', 'GPU'], help="Cycles device selection")
    parser.add_argument(
        '--cycles_compute_type',
        type=str,
        default='CUDA',
        choices=['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'],
        help="Cycles GPU compute backend",
    )
    parser.add_argument('--exposure', type=float, default=0.0, help="Color management exposure")
    parser.add_argument('--gamma', type=float, default=1.0, help="Color management gamma")
    parser.add_argument('--view_transform', type=str, default='AgX', help="Color management view transform")
    parser.add_argument('--look', type=str, default='None', help="Color management look")

    # Optional arguments - lighting
    parser.add_argument('--no_lights', action='store_true', help="Disable all lights")
    parser.add_argument('--point_light_pos', nargs=3, type=float, default=[5, -5, 5], help="Point light location X Y Z")
    parser.add_argument('--point_light_energy', type=float, default=800.0, help="Point light energy")
    parser.add_argument('--point_light_color', nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Point light RGB")
    parser.add_argument('--point_light_soft_size', type=float, default=0.0, help="Point light shadow soft size")
    parser.add_argument('--area_light_color', nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Area light RGB")
    parser.add_argument('--area_light_pos', nargs=3, type=float, default=[-5, -5, 8], help="Area light location X Y Z")
    parser.add_argument('--area_light_rot', nargs=3, type=float, default=[math.radians(60), 0.0, math.radians(45)], help="Area light rotation Euler radians")
    parser.add_argument('--area_light_rot_deg', nargs=3, type=float, default=None, help="Area light rotation Euler degrees")
    parser.add_argument('--area_light_energy', type=float, default=1500.0, help="Area light energy")
    parser.add_argument('--area_light_size', type=float, default=5.0, help="Area light size")
    parser.add_argument('--area_light_size_y', type=float, default=None, help="Area light size Y")
    parser.add_argument('--area_light_spread', type=float, default=1.0, help="Area light spread")
    parser.add_argument('--lights_target_center', action='store_true', help="Rotate area light to face point cloud center")
    
    parser.add_argument('--sun_light_energy', type=float, default=0.0, help="Sun light energy (0=off)")
    parser.add_argument('--sun_light_color', nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Sun light RGB")
    parser.add_argument('--sun_light_rot', nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Sun light rotation Euler radians")
    parser.add_argument('--sun_light_rot_deg', nargs=3, type=float, default=None, help="Sun light rotation Euler degrees")
    parser.add_argument('--sun_light_angle', type=float, default=0.00918, help="Sun light angle in radians (default 0.526 deg)")

    # Optional arguments - ground shadow
    parser.add_argument('--ground', action='store_true', help="Enable a ground plane shadow")
    parser.add_argument('--ground_point', nargs=3, type=float, default=None, help="Ground point X Y Z")
    parser.add_argument('--ground_normal', nargs=3, type=float, default=[0.0, 0.0, 1.0], help="Ground normal X Y Z")
    parser.add_argument('--ground_offset', type=float, default=0.08, help="Auto ground offset")
    parser.add_argument('--ground_size', type=float, default=0.0, help="Ground plane size (0=auto)")
    parser.add_argument('--no_shadow_catcher', action='store_true', help="Disable shadow-catcher ground")

    # Optional arguments - material
    parser.add_argument('--roughness', type=float, default=0.5, help="Principled roughness")
    parser.add_argument('--specular', type=float, default=0.3, help="Principled specular")
    parser.add_argument('--metallic', type=float, default=0.0, help="Principled metallic")
    parser.add_argument('--emission_strength', type=float, default=0.001, help="Principled emission strength")
    parser.add_argument('--ao', action='store_true', help="Enable Ambient Occlusion")
    parser.add_argument('--save_blend', type=str, default="", help="Optional .blend path to save")

    args = parser.parse_args(args)

    # --- Begin Blender operations ---
    clean_scene()

    imported_objects = []

    # Import and set up point cloud/Mesh
    for input_path in args.input:
        print(f"Importing {input_path}...")
        
        # Deselect all before import to ensure correct selection
        bpy.ops.object.select_all(action='DESELECT')

        obj = import_and_process_ply(
            input_path,
            args.radius,
            icosphere_subdivisions=args.ico_subdiv,
            original_col_name=args.col_name,
            color_override_rgba=args.color_override,
            material_roughness=args.roughness,
            material_specular=args.specular,
            material_metallic=args.metallic,
            material_emission_strength=args.emission_strength,
        )
        imported_objects.append(obj)

        apply_transform = args.swap_yz or (args.obj_scale is not None) or (args.obj_rot_deg is not None) or (args.obj_location is not None)
        if apply_transform:
            import mathutils

            swap_matrix = mathutils.Matrix.Identity(4)
            if args.swap_yz:
                print(f"Swapping Y and Z coordinates for {obj.name}...")
                swap_matrix = mathutils.Matrix.Rotation(math.radians(-90), 4, "X")

            scale_matrix = mathutils.Matrix.Identity(4)
            if args.obj_scale is not None:
                print(f"Applying custom scale to {obj.name}: {args.obj_scale}...")
                scale_matrix = mathutils.Matrix.Diagonal((*args.obj_scale, 1.0))

            rot_matrix = mathutils.Matrix.Identity(4)
            if args.obj_rot_deg is not None:
                print(f"Applying custom rotation to {obj.name}: {args.obj_rot_deg} degrees...")
                rad_rot = [math.radians(d) for d in args.obj_rot_deg]
                euler_rot = mathutils.Euler(rad_rot, "XYZ")
                rot_matrix = euler_rot.to_matrix().to_4x4()

            translation_matrix = mathutils.Matrix.Identity(4)
            if args.obj_location is not None:
                print(f"Applying custom location to {obj.name}: {args.obj_location}...")
                translation_matrix = mathutils.Matrix.Translation(args.obj_location)

            combined_matrix = translation_matrix @ rot_matrix @ scale_matrix @ swap_matrix
            obj.data.transform(combined_matrix)
            obj.data.update()

    if not imported_objects:
        print("No objects imported.")
        return

    # Calculate global bounds
    from mathutils import Vector
    global_min = Vector((float('inf'), float('inf'), float('inf')))
    global_max = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in imported_objects:
        min_c, max_c = _get_object_bounds(obj)
        global_min.x = min(global_min.x, min_c[0])
        global_min.y = min(global_min.y, min_c[1])
        global_min.z = min(global_min.z, min_c[2])
        global_max.x = max(global_max.x, max_c[0])
        global_max.y = max(global_max.y, max_c[1])
        global_max.z = max(global_max.z, max_c[2])

    center_vec = (global_min + global_max) * 0.5
    radius_val = (global_max - global_min).length * 0.5

    obj_center = (float(center_vec.x), float(center_vec.y), float(center_vec.z))
    obj_radius = float(radius_val)
    obj_min = (float(global_min.x), float(global_min.y), float(global_min.z))

    transparent = args.background_rgb is None
    setup_world(background_rgb=args.background_rgb, world_strength=args.world_strength, transparent=transparent)

    if args.ground:
        _add_ground_plane(
            center=obj_center,
            radius=obj_radius,
            obj_min=obj_min,
            ground_normal=args.ground_normal,
            ground_point=args.ground_point,
            offset=args.ground_offset,
            size=args.ground_size,
            shadow_catcher=not args.no_shadow_catcher,
        )
    
    # Set up the camera
    cam_pos = args.cam_pos
    cam_rot = args.cam_rot
    clip_start = None
    clip_end = None
    ortho_scale = args.ortho_scale

    if args.cam_rot_deg is not None:
        cam_rot = _degrees_to_radians(args.cam_rot_deg)

    cam_target = args.cam_target
    if args.look_at_center:
        cam_target = obj_center

    if args.auto_frame:
        cam_pos, cam_rot, clip_start, clip_end = _auto_frame_camera(
            obj_center,
            obj_radius,
            args.fov,
            args.auto_cam_dir,
            margin=args.frame_margin,
            distance_scale=args.auto_distance_scale,
        )
        cam_target = obj_center
        if args.orthographic:
            ortho_scale = float(obj_radius) * 2.0 * float(args.frame_margin)

    if cam_target is not None:
        cam_rot = _look_at_euler(cam_pos, cam_target)

    cam_obj = setup_camera(
        cam_pos, 
        cam_rot, 
        fov=args.fov, 
        is_ortho=args.orthographic, 
        ortho_scale=ortho_scale
    )
    if clip_start is not None:
        cam_obj.data.clip_start = float(clip_start)
    if clip_end is not None:
        cam_obj.data.clip_end = float(clip_end)
    
    if not args.no_lights:
        point_light_pos = args.point_light_pos
        area_light_pos = args.area_light_pos
        area_light_size = args.area_light_size
        area_light_size_y = args.area_light_size_y
        point_light_soft_size = float(args.point_light_soft_size)

        if args.auto_frame and args.auto_lights:
            r = max(float(obj_radius), 1e-6)
            point_light_pos = [obj_center[0] + 2.0 * r, obj_center[1] - 2.0 * r, obj_center[2] + 2.0 * r]
            area_light_pos = [obj_center[0] - 2.5 * r, obj_center[1] - 1.5 * r, obj_center[2] + 3.0 * r]
            area_light_size = max(4.0 * r, 0.5)
            if area_light_size_y is not None:
                area_light_size_y = max(float(area_light_size_y), 0.5)
            if point_light_soft_size <= 0:
                point_light_soft_size = 0.5 * r

        area_rot = args.area_light_rot
        if args.area_light_rot_deg is not None:
            area_rot = _degrees_to_radians(args.area_light_rot_deg)
        if args.lights_target_center:
            area_rot = _look_at_euler(area_light_pos, obj_center)

        setup_light(
            area_light_pos,
            energy=args.area_light_energy,
            type="AREA",
            rotation_euler=area_rot,
            color=tuple(args.area_light_color),
            size=area_light_size,
            size_y=area_light_size_y,
            spread=args.area_light_spread,
        )
        setup_light(
            point_light_pos,
            energy=args.point_light_energy,
            type="POINT",
            color=tuple(args.point_light_color),
            shadow_soft_size=point_light_soft_size,
        )

        if args.sun_light_energy > 0:
            sun_rot = args.sun_light_rot
            if args.sun_light_rot_deg is not None:
                sun_rot = _degrees_to_radians(args.sun_light_rot_deg)
            
            setup_light(
                (0, 0, 10),
                energy=args.sun_light_energy,
                type="SUN",
                rotation_euler=sun_rot,
                color=tuple(args.sun_light_color),
                angle=args.sun_light_angle,
            )

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.filepath = args.output

    scene.view_settings.exposure = float(args.exposure)
    scene.view_settings.gamma = float(args.gamma)
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = args.view_transform
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = args.look

    output_path = Path(args.output)
    inferred_format = "PNG"
    suffix = output_path.suffix.lower()
    if suffix == ".exr":
        inferred_format = "OPEN_EXR"
    elif suffix in {".jpg", ".jpeg"}:
        inferred_format = "JPEG"
    elif suffix == ".png":
        inferred_format = "PNG"

    file_format = args.file_format or inferred_format
    scene.render.image_settings.file_format = file_format
    if file_format == "PNG":
        scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    
    if args.engine == 'CYCLES':
        scene.cycles.samples = args.samples

        if args.denoise:
            if hasattr(scene.cycles, "use_denoising"):
                scene.cycles.use_denoising = True
            if hasattr(bpy.context.view_layer, "cycles") and hasattr(bpy.context.view_layer.cycles, "use_denoising"):
                bpy.context.view_layer.cycles.use_denoising = True

        if args.cycles_device != "CPU":
            try:
                prefs = bpy.context.preferences
                cprefs = prefs.addons["cycles"].preferences
                
                # 1. Set compute device type
                cprefs.compute_device_type = args.cycles_compute_type
                
                # 2. Refresh devices (crucial for Blender to see them)
                if hasattr(cprefs, "get_devices"):
                    cprefs.get_devices()
                
                # 3. Enable devices
                enabled_count = 0
                for device in cprefs.devices:
                    if device.type == args.cycles_compute_type:
                        device.use = True
                        enabled_count += 1
                        print(f"Enabled Cycles device: {device.name}")
                    # You might want to keep CPU enabled or not, usually for hybrid rendering we might keep it
                    # But for strict GPU rendering, we often only enable GPU.
                    # Here we only explicitly enable the matching type.

                if enabled_count > 0:
                    scene.cycles.device = "GPU"
                    print(f"Cycles set to GPU ({args.cycles_compute_type}) with {enabled_count} devices.")
                else:
                    print(f"Warning: No {args.cycles_compute_type} devices found. Falling back to CPU.")
                    scene.cycles.device = "CPU"
            except Exception as e:
                print(f"Failed to enable GPU: {e}. Falling back to CPU.")
                scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "CPU"
    elif args.engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = args.samples
    
    print(f"Rendering to {args.output}...")
    bpy.ops.render.render(write_still=True)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    print("Done!")

if __name__ == "__main__":
    main()
