import bpy
import sys
import argparse
import math
from pathlib import Path

def clean_scene():
    """Clean all objects in the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera(location, rotation_euler, fov=50):
    """Setup camera"""
    bpy.ops.object.camera_add(location=location, rotation=rotation_euler)
    cam = bpy.context.object
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
):
    """Setup light"""
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


def _look_at_euler(location, target, world_up=(0.0, 0.0, 1.0)):
    from mathutils import Matrix, Vector

    direction = Vector(target) - Vector(location)
    if direction.length == 0:
        return (0.0, 0.0, 0.0)

    forward = direction.normalized()

    up = Vector(world_up)
    if up.length == 0:
        up = Vector((0.0, 0.0, 1.0))
    up.normalize()

    right = forward.cross(up)
    if right.length < 1e-8:
        alt_up = Vector((0.0, 1.0, 0.0))
        if abs(forward.dot(alt_up)) > 0.99:
            alt_up = Vector((1.0, 0.0, 0.0))
        right = forward.cross(alt_up)

    right.normalize()
    up2 = right.cross(forward).normalized()

    rot_m = Matrix((right, up2, -forward)).transposed()
    return rot_m.to_euler()


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


def _auto_frame_camera(center, radius, fov_deg, cam_dir, world_up=(0.0, 0.0, 1.0), margin=1.2, distance_scale=1.0):
    from mathutils import Vector

    radius = max(float(radius), 1e-6)
    half_angle = math.radians(float(fov_deg)) * 0.5
    dist = (radius / max(math.tan(half_angle), 1e-6)) * float(margin) * float(distance_scale)

    cam_dir_n = Vector(_normalize_vec3(cam_dir))
    cam_pos = Vector(center) - cam_dir_n * dist
    cam_rot = _look_at_euler(cam_pos, center, world_up=world_up)

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


def create_shader_material(
    mat_name="Mesh_Material",
    col_name="Col",
    roughness=0.4,
    specular=0.5,
    metallic=0.0,
    emission_strength=0.0,
    base_color=None,
    use_vertex_color=True,
):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Get Principled BSDF and Output
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
    
    if use_vertex_color:
        attr_node = nodes.new('ShaderNodeAttribute')
        attr_node.attribute_type = 'GEOMETRY'
        attr_node.attribute_name = col_name
        attr_node.location = (-300, 200)
        if "Base Color" in bsdf.inputs:
            links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
        if emission_strength and emission_strength > 0:
            if "Emission Color" in bsdf.inputs:
                links.new(attr_node.outputs["Color"], bsdf.inputs["Emission Color"])
            elif "Emission" in bsdf.inputs:
                links.new(attr_node.outputs["Color"], bsdf.inputs["Emission"])
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = float(emission_strength)
    else:
        if base_color is None:
            base_color = (0.8, 0.8, 0.8)
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0)

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(specular)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(metallic)

    return mat


def create_overlay_material(
    mat_name="Overlay_Material",
    col_name="Col",
    emission_strength=8.0,
):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'NONE'
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (300, 0)

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = col_name
    attr.location = (-500, 0)

    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    transparent.location = (-200, -120)

    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = (-200, 120)
    if "Color" in emission.inputs:
        links.new(attr.outputs["Color"], emission.inputs["Color"])
    if "Strength" in emission.inputs:
        emission.inputs["Strength"].default_value = float(emission_strength)

    mix = nodes.new(type="ShaderNodeMixShader")
    mix.location = (80, 0)
    if "Fac" in mix.inputs:
        links.new(attr.outputs["Alpha"], mix.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])

    return mat


def import_and_process_mesh(
    filepath,
    col_name="Col",
    material_roughness=0.4,
    material_specular=0.5,
    material_metallic=0.0,
    material_emission_strength=0.0,
    material_base_color=None,
    material_use_vertex_color=True,
    shade_smooth=True,
    subdivision_levels=0,
):
    path = Path(filepath)
    ext = path.suffix.lower()
    
    # Import
    if ext == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=str(path))
        except AttributeError:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(path))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        # Try generic import or fallback
        try:
            bpy.ops.import_scene.obj(filepath=str(path))
        except Exception as e:
            print(f"Warning: Unknown extension {ext}, trying OBJ import as fallback. Error: {e}")
            pass
        
    selected = list(bpy.context.selected_objects)
    obj = bpy.context.view_layer.objects.active if bpy.context.view_layer.objects.active in selected else (selected[0] if selected else None)
    if obj is None:
        raise RuntimeError(f"Failed to import mesh: {filepath}")
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if int(subdivision_levels) > 0:
        mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
        lvl = int(subdivision_levels)
        mod.levels = lvl
        mod.render_levels = lvl

    # Smooth shading
    if shade_smooth:
        bpy.ops.object.shade_smooth()
    
    # Create Material
    mat = create_shader_material(
        col_name=col_name,
        roughness=material_roughness,
        specular=material_specular,
        metallic=material_metallic,
        emission_strength=material_emission_strength,
        base_color=material_base_color,
        use_vertex_color=material_use_vertex_color,
    )
    
    # Apply Material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj


def main():
    # --- Handle Command Line Args ---
    if "--" not in sys.argv:
        args = []
    else:
        args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Render Mesh (PLY/OBJ) with Blender")
    
    # Required
    parser.add_argument('--input', type=str, required=True, help="Input mesh file path (.ply, .obj)")
    parser.add_argument('--output', type=str, required=True, help="Output image path (e.g. result.png)")
    
    # Optional - Scene
    parser.add_argument('--col_name', type=str, default="Col", help="Name of color attribute in Mesh")
    parser.add_argument('--overlay', type=str, action="append", default=[], help="Overlay mesh path(s) rendered on top")
    parser.add_argument('--overlay_emission', type=float, default=8.0, help="Overlay emission strength")
    parser.add_argument('--overlay_scale', type=float, default=1.001, help="Overlay uniform scale factor")
    parser.add_argument('--shade_flat', action='store_true', help="Use flat shading instead of smooth")
    parser.add_argument('--subdivision', type=int, default=0, help="Subdivision surface levels (0 disables)")
    parser.add_argument('--obj_rot_deg', nargs=3, type=float, default=None, help="Custom object rotation in degrees (Euler X Y Z)")
    parser.add_argument('--base_color', nargs=3, type=float, default=None, help="If set, ignore vertex colors and use base RGB")

    # Optional - Background/World
    parser.add_argument(
        '--background_rgb',
        nargs=3,
        type=float,
        default=None,
        help="Opaque background RGB (0..1). If omitted, render with transparent background.",
    )
    parser.add_argument('--world_strength', type=float, default=0.0, help="World background strength")
    
    # Optional - Camera
    parser.add_argument('--cam_pos', nargs=3, type=float, default=[0, -5, 1], help="Camera location X Y Z")
    parser.add_argument('--cam_rot', nargs=3, type=float, default=[math.radians(80), 0, 0], help="Camera rotation Euler X Y Z (in radians)")
    parser.add_argument('--cam_rot_deg', nargs=3, type=float, default=None, help="Camera rotation Euler X Y Z (in degrees)")
    parser.add_argument('--cam_target', nargs=3, type=float, default=None, help="Look-at target X Y Z")
    parser.add_argument('--look_at_center', action='store_true', help="Make camera look at mesh center")
    parser.add_argument('--world_up', nargs=3, type=float, default=[0.0, 0.0, 1.0], help="World up direction X Y Z")
    parser.add_argument('--fov', type=float, default=50.0, help="Camera field of view (degrees)")
    parser.add_argument('--auto_frame', action='store_true', help="Auto frame camera and lights using bounding box")
    parser.add_argument('--frame_margin', type=float, default=1.2, help="Margin for auto framing")
    parser.add_argument('--auto_cam_dir', nargs=3, type=float, default=[0.0, -1.0, 0.25], help="Auto camera direction")
    parser.add_argument('--auto_distance_scale', type=float, default=1.0, help="Auto camera distance scale")
    parser.add_argument('--auto_lights', action='store_true', help="Auto place lights when auto framing")
    
    # Optional - Render
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
    parser.add_argument('--view_transform', type=str, default='Filmic', help="Color management view transform")
    parser.add_argument('--look', type=str, default='None', help="Color management look")

    # Optional - Lights
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
    parser.add_argument('--lights_target_center', action='store_true', help="Rotate area light to face mesh center")

    # Optional - Ground Shadow
    parser.add_argument('--ground', action='store_true', help="Enable a ground plane shadow")
    parser.add_argument('--ground_point', nargs=3, type=float, default=None, help="Ground point X Y Z")

    # Material params
    parser.add_argument('--roughness', type=float, default=0.4, help="Material roughness")
    parser.add_argument('--specular', type=float, default=0.5, help="Material specular")
    parser.add_argument('--metallic', type=float, default=0.0, help="Material metallic")
    parser.add_argument('--emission', type=float, default=0.0, help="Material emission strength")

    args = parser.parse_args(args)

    clean_scene()
    
    # 1. Setup World
    setup_world(background_rgb=args.background_rgb, world_strength=args.world_strength, transparent=(args.background_rgb is None))

    # 2. Import and Process Mesh
    mesh_obj = import_and_process_mesh(
        args.input,
        col_name=args.col_name,
        material_roughness=args.roughness,
        material_specular=args.specular,
        material_metallic=args.metallic,
        material_emission_strength=args.emission,
        material_base_color=args.base_color,
        material_use_vertex_color=(args.base_color is None),
        shade_smooth=(not args.shade_flat),
        subdivision_levels=args.subdivision,
    )

    if args.obj_rot_deg is not None:
        mesh_obj.rotation_euler = _degrees_to_radians(args.obj_rot_deg)

    # 3. Calculations for Camera/Lights
    center, radius = _get_object_center_and_radius(mesh_obj)
    obj_min, obj_max = _get_object_bounds(mesh_obj)

    overlay_objs = []
    if args.overlay:
        overlay_mat = create_overlay_material(
            col_name=args.col_name,
            emission_strength=args.overlay_emission,
        )
        for overlay_path in args.overlay:
            ov = import_and_process_mesh(
                overlay_path,
                col_name=args.col_name,
                material_roughness=1.0,
                material_specular=0.0,
                material_metallic=0.0,
                material_emission_strength=0.0,
                material_base_color=None,
                material_use_vertex_color=True,
                shade_smooth=(not args.shade_flat),
                subdivision_levels=args.subdivision,
            )
            if args.obj_rot_deg is not None:
                ov.rotation_euler = _degrees_to_radians(args.obj_rot_deg)
            s = float(args.overlay_scale)
            ov.scale = (s, s, s)
            if ov.data.materials:
                ov.data.materials[0] = overlay_mat
            else:
                ov.data.materials.append(overlay_mat)
            overlay_objs.append(ov)

    # 4. Camera
    cam_loc = args.cam_pos
    cam_rot = args.cam_rot
    if args.cam_rot_deg is not None:
        cam_rot = _degrees_to_radians(args.cam_rot_deg)

    # Auto Frame overrides
    if args.auto_frame:
        cam_loc, cam_rot, clip_start, clip_end = _auto_frame_camera(
            center,
            radius,
            args.fov,
            args.auto_cam_dir,
            world_up=args.world_up,
            margin=args.frame_margin,
            distance_scale=args.auto_distance_scale,
        )
        cam = setup_camera(cam_loc, cam_rot, fov=args.fov)
        cam.data.clip_start = clip_start
        cam.data.clip_end = clip_end
    else:
        # Manual
        if args.cam_target is not None:
            cam_rot = _look_at_euler(cam_loc, args.cam_target, world_up=args.world_up)
        elif args.look_at_center:
            cam_rot = _look_at_euler(cam_loc, center, world_up=args.world_up)
        cam = setup_camera(cam_loc, cam_rot, fov=args.fov)

    # 5. Lights
    if not args.no_lights:
        if args.auto_frame and args.auto_lights:
            # Simple auto lighting relative to camera
            from mathutils import Vector, Euler
            
            # Key light (Area)
            cam_obj = bpy.context.scene.camera
            cam_dir = cam_obj.matrix_world.to_quaternion() @ Vector((0, 0, -1))
            cam_right = cam_obj.matrix_world.to_quaternion() @ Vector((1, 0, 0))
            cam_up = cam_obj.matrix_world.to_quaternion() @ Vector((0, 1, 0))
            
            # 45 deg right, 45 deg up
            light_pos = Vector(cam_loc) + (cam_right * radius * 2.0) + (cam_up * radius * 2.0)
            
            # Look at center
            light_rot = _look_at_euler(light_pos, center, world_up=args.world_up)
            
            setup_light(
                location=light_pos,
                rotation_euler=light_rot,
                type="AREA",
                energy=args.area_light_energy,
                size=args.area_light_size,
                size_y=args.area_light_size_y,
                spread=args.area_light_spread,
                color=args.area_light_color,
            )
            
            # Fill light (Point) - opposite side
            fill_pos = Vector(cam_loc) - (cam_right * radius * 2.0) + (cam_up * radius * 1.0)
            setup_light(
                location=fill_pos,
                type="POINT",
                energy=args.point_light_energy * 0.5,
                color=args.point_light_color,
            )
            
        else:
            # Manual Lights
            setup_light(
                location=args.point_light_pos,
                type="POINT",
                energy=args.point_light_energy,
                color=args.point_light_color,
                shadow_soft_size=args.point_light_soft_size,
            )

            area_rot = args.area_light_rot
            if args.area_light_rot_deg is not None:
                area_rot = _degrees_to_radians(args.area_light_rot_deg)
            elif args.lights_target_center:
                area_rot = _look_at_euler(args.area_light_pos, center, world_up=args.world_up)

            setup_light(
                location=args.area_light_pos,
                rotation_euler=area_rot,
                type="AREA",
                energy=args.area_light_energy,
                color=args.area_light_color,
                size=args.area_light_size,
                size_y=args.area_light_size_y,
                spread=args.area_light_spread,
            )

    # 6. Ground
    if args.ground:
        _add_ground_plane(
            center,
            radius,
            obj_min,
            ground_normal=(0, 0, 1),
            ground_point=args.ground_point,
            offset=0.0,
            shadow_catcher=True,
        )

    # 7. Render Settings
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    
    if args.engine == 'CYCLES':
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = args.denoise
        
        # Set compute device type
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = args.cycles_compute_type
        cycles_prefs.get_devices()
        
        # Handle 'AUTO' device selection
        if args.cycles_device == 'AUTO':
            # Use GPU if any non-CPU device is available
            has_gpu = any(d.type != 'CPU' for d in cycles_prefs.devices)
            scene.cycles.device = 'GPU' if has_gpu else 'CPU'
        else:
            scene.cycles.device = args.cycles_device
            
        if scene.cycles.device == 'GPU':
            for device in cycles_prefs.devices:
                if device.type != 'CPU':
                    device.use = True
                else:
                    device.use = False
    
    # Color Management
    scene.view_settings.view_transform = args.view_transform
    scene.view_settings.look = args.look
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = args.gamma

    # Output
    transparent = (args.background_rgb is None)
    suffix = Path(args.output).suffix.lower()
    inferred_format = ""
    if suffix == ".jpg" or suffix == ".jpeg":
        inferred_format = "JPEG"
    elif suffix == ".png":
        inferred_format = "PNG"
    file_format = args.file_format or inferred_format
    if file_format:
        scene.render.image_settings.file_format = file_format
    if (scene.render.image_settings.file_format == "PNG") and hasattr(scene.render.image_settings, "color_mode"):
        scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    
    scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
    print(f"Render saved to {args.output}")

if __name__ == "__main__":
    main()
