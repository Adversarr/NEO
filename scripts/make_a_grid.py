import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import sys
import json
import os
import trimesh.transformations as tf

def normalize_feature(feat, symmetric=True):
    """
    Normalize feature vector to [0, 1] for colormap application.
    If symmetric is True, 0 maps to 0.5 (for diverging colormaps).
    """
    if symmetric:
        vmax = np.max(np.abs(feat))
        if vmax == 0:
            return np.full_like(feat, 0.5)
        # Map [-vmax, vmax] to [0, 1]
        return 0.5 + 0.5 * (feat / vmax)
    else:
        vmin, vmax = np.min(feat), np.max(feat)
        if vmin == vmax:
            return np.zeros_like(feat)
        return (feat - vmin) / (vmax - vmin)

def load_rotation_override(path):
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def get_transform_for_case(case_name, overrides):
    # Try exact match
    if case_name in overrides:
        data = overrides[case_name]
        return data.get("rotate", [0, 0, 0]), data.get("scale", 1.0)
    
    # Try matching by stripping common suffixes like "-Uniform"
    # Adjust this logic based on how keys are stored in json vs folder names
    # Example: folder "armadillo-Uniform" -> key "armadillo"
    if "-Uniform" in case_name or '-NonUniform-0.4' in case_name:
        base_name = case_name.replace("-Uniform", "").replace('-NonUniform-0.4', '')
        if base_name in overrides:
            data = overrides[base_name]
            return data.get("rotate", [0, 0, 0]), data.get("scale", 1.0)
            
    return [0, 0, 0], 1.0

def main():
    parser = argparse.ArgumentParser(description="Create a grid of point clouds (Rows: Models, Cols: Base Color + Eigenfunctions).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing case subdirectories")
    parser.add_argument("--output", type=str, required=True, help="Path to the output PLY file")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4], help="Indices of eigenvectors to visualize in subsequent columns")
    parser.add_argument("--spacing_x", type=float, default=1.2, help="Spacing between different meshes (X axis)")
    parser.add_argument("--spacing_z", type=float, default=1.2, help="Spacing between eigenfunctions (Z axis)")
    parser.add_argument("--rotate_override", type=str, default="/home/adversarr/Repo/g2pt/rotate_overide.json", help="Path to rotation override JSON")
    parser.add_argument("--base_color", type=float, nargs=3, default=[0.22, 0.34, 0.48], help="RGB color for the first column (0-1)")
    parser.add_argument("--colormap", type=str, default="coolwarm", help="Colormap to use")
    parser.add_argument("--symmetric", action="store_true", default=True, help="Use symmetric normalization")
    parser.add_argument("--downsample", type=int, nargs='?', const=2048, default=None, help="Downsample point clouds to N points (default: 2048 if flag provided)")
    parser.add_argument("--add_rotate", type=float, nargs=3, default=[0, 0, 0], help="Additive rotation in degrees (X, Y, Z)")
    parser.add_argument("--global-scale", type=float, default=1.0, help="Global scale factor applied after normalization")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    rotate_override_path = Path(args.rotate_override)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)
        
    overrides = load_rotation_override(rotate_override_path)
    print(overrides)
    
    # Find valid case directories
    case_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and (d / "input" / "points.npy").exists()])
    
    if not case_dirs:
        print("No valid case directories found.")
        sys.exit(1)
        
    print(f"Found {len(case_dirs)} cases.")
    
    try:
        cmap = plt.get_cmap(args.colormap)
    except ValueError:
        try:
            import cmcrameri
            cmap = plt.get_cmap(f"cmc.{args.colormap}")
        except (ValueError, ImportError):
             print(f"Warning: Colormap '{args.colormap}' not found. Falling back to 'coolwarm'.")
             cmap = plt.get_cmap("coolwarm")

    all_meshes = []
    
    base_color_uint8 = (np.array(args.base_color) * 255).astype(np.uint8)
    if len(base_color_uint8) == 3:
        base_color_uint8 = np.append(base_color_uint8, 255) # Add alpha
    
    for row_idx, case_dir in enumerate(case_dirs):
        case_name = case_dir.name
        
        points_path = case_dir / "input" / "points.npy"
        faces_path = case_dir / "input" / "faces.npy"
        pc_evecs_path = case_dir / "inferred" / "pc_evec.npy" if 'Airplane' in str(case_dir) else case_dir / "inferred" / "net_evec.npy"
        
        points = np.load(points_path)
        faces = None
        if faces_path.exists():
             faces = np.load(faces_path)

        # Downsample if requested
        downsample_idx = None
        if args.downsample is not None and points.shape[0] > args.downsample:
            # Use fixed seed for reproducibility
            np.random.seed(42) 
            downsample_idx = np.random.choice(points.shape[0], args.downsample, replace=False)
            points = points[downsample_idx]
            if faces is not None:
                print(f"  Warning: Downsampling enabled, discarding faces for {case_name}.")
                faces = None


        # Normalize (bounding box)
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = (max_coords - min_coords).max()
        if scale > 0:
            points = (points - center) / scale
        
        # Get transform overrides
        rotation, scale_override = get_transform_for_case(case_name, overrides)

        # Apply global scale and override scale
        points = points * args.global_scale * scale_override

        # Apply rotation
        rotation = np.array(rotation)
        if any(r != 0 for r in rotation):
            # print(f"  Applying rotation {rotation}")
            matrix = tf.euler_matrix(*(np.radians(rotation)), axes='sxyz')
            points = points @ matrix[:3, :3].T
        print(f"Processing row {row_idx}: {case_name}, rot={rotation}, scale={scale_override}")

        g_rotate = np.array(args.add_rotate)
        if any(r != 0 for r in g_rotate):
            matrix = tf.euler_matrix(*(np.radians(g_rotate)), axes='sxyz')
            points = points @ matrix[:3, :3].T

        # --- Variant 0: Base Color (at Z=0) ---
        # Mesh index determines X position
        offset_x = row_idx * args.spacing_x
        offset_base = np.array([offset_x, 0, 0])
        
        shifted_points_0 = points + offset_base
        colors_0 = np.tile(base_color_uint8, (len(points), 1))
        
        if faces is not None:
             mesh_0 = trimesh.Trimesh(vertices=shifted_points_0, faces=faces, process=False)
             mesh_0.visual.vertex_colors = colors_0
             all_meshes.append(mesh_0)
        else:
             pcd_0 = trimesh.PointCloud(vertices=shifted_points_0, colors=colors_0)
             all_meshes.append(pcd_0)
        
        # --- Variants 1..N: Eigenfunctions (along Z axis) ---
        if pc_evecs_path.exists():
            pc_evecs = np.load(pc_evecs_path)
            
            if downsample_idx is not None:
                pc_evecs = pc_evecs[downsample_idx]

            for col_idx, k in enumerate(args.k):
                # Variant index determines Z position
                variant_idx = col_idx + 1
                offset_z = variant_idx * args.spacing_z
                offset = np.array([offset_x, 0, offset_z])
                
                if k < pc_evecs.shape[1]:
                    feat = pc_evecs[:, k]
                    norm_feat = normalize_feature(feat, symmetric=args.symmetric)
                    colors_rgba = cmap(norm_feat)
                    colors_uint8 = (colors_rgba * 255).astype(np.uint8)
                    
                    shifted_points = points + offset
                    
                    if faces is not None:
                        mesh_k = trimesh.Trimesh(vertices=shifted_points, faces=faces, process=False)
                        mesh_k.visual.vertex_colors = colors_uint8
                        all_meshes.append(mesh_k)
                    else:
                        pcd_k = trimesh.PointCloud(vertices=shifted_points, colors=colors_uint8)
                        all_meshes.append(pcd_k)
                else:
                    print(f"  Warning: Eigenvector index {k} out of bounds (max {pc_evecs.shape[1]-1}). Skipping.")
        else:
            print(f"  Warning: {pc_evecs_path} not found. Skipping eigenvector columns.")

    if not all_meshes:
        print("No meshes/points generated.")
        sys.exit(1)
        
    print(f"Merging {len(all_meshes)} objects...")
    
    # Manual concatenation to handle mixed Trimesh/PointCloud objects
    all_vertices = []
    all_colors = []
    all_faces = []
    vertex_offset = 0
    
    for obj in all_meshes:
        v = obj.vertices
        # Ensure colors are available (N, 4)
        c = obj.visual.vertex_colors
        
        all_vertices.append(v)
        all_colors.append(c)
        
        # Check if object has faces
        if hasattr(obj, 'faces') and obj.faces is not None and len(obj.faces) > 0:
            f = obj.faces + vertex_offset
            all_faces.append(f)
            
        vertex_offset += len(v)
        
    combined_vertices = np.vstack(all_vertices)
    combined_colors = np.vstack(all_colors)
    
    if all_faces:
        combined_faces = np.vstack(all_faces)
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
    else:
        combined_mesh = trimesh.PointCloud(vertices=combined_vertices)
        
    combined_mesh.visual.vertex_colors = combined_colors
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {output_path}...")
    combined_mesh.export(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
