import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import sys
import cmcrameri
import seaborn as sns # type: ignore

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

def parse_feature_arg(arg):
    """
    Parse feature argument which can be "int", "pc:int", or "net:int".
    Returns (source_type, index) where source_type is 'pc' or 'net'.
    """
    if ":" in arg:
        parts = arg.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid feature format: {arg}. Expected 'source:index' or just 'index'.")
        source, idx = parts[0], parts[1]
        if source not in ["pc", "net"]:
            raise ValueError(f"Invalid source: {source}. Expected 'pc' or 'net'.")
        return source, int(idx)
    else:
        return "pc", int(arg)

def main():
    parser = argparse.ArgumentParser(description="Duplicate point cloud with different colormaps arranged in a row.")
    parser.add_argument("--case", type=str, required=True, help="Path to the case directory (output of infer.py)")
    parser.add_argument("--output", type=str, required=True, help="Path to the output PLY file")
    parser.add_argument("--features", type=str, nargs="+", default=["0", "1", "2", "3"], 
                        help="Indices of features/eigenvectors to visualize. Format: 'idx', 'pc:idx', or 'net:idx'. Default source is 'pc'.")
    parser.add_argument("--direction", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Direction vector for the row")
    parser.add_argument("--spacing", type=float, default=1.2, help="Spacing between copies")
    parser.add_argument("--colormap", type=str, default="coolwarm", help="Colormap to use (default: coolwarm)")
    parser.add_argument("--symmetric", action="store_true", default=True, help="Use symmetric normalization for diverging colormaps (default: True)")
    parser.add_argument("--no-symmetric", action="store_false", dest="symmetric", help="Disable symmetric normalization")
    parser.add_argument("--rotate", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Rotation in degrees (x, y, z) to apply to input mesh")
    
    args = parser.parse_args()
    
    case_dir = Path(args.case)
    points_path = case_dir / "input" / "points.npy"
    pc_evecs_path = case_dir / "inferred" / "pc_evec.npy"
    net_evecs_path = case_dir / "inferred" / "net_evec.npy"
    
    if not points_path.exists():
        print(f"Error: {points_path} does not exist.")
        sys.exit(1)
    if not pc_evecs_path.exists():
        print(f"Error: {pc_evecs_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading points from {points_path}...")
    points = np.load(points_path) # (N, 3)
    print(f"Points shape: {points.shape}")

    if any(r != 0 for r in args.rotate):
        import trimesh.transformations as tf
        print(f"Rotating input mesh by {args.rotate} degrees...")
        matrix = tf.euler_matrix(*(np.radians(args.rotate)), axes='sxyz')
        # Apply rotation (N, 3) @ (3, 3)
        points = points @ matrix[:3, :3].T

    # Determine which files to load
    parsed_features = []
    need_pc = False
    need_net = False
    
    for f_arg in args.features:
        try:
            source, idx = parse_feature_arg(f_arg)
            parsed_features.append((source, idx))
            if source == "pc":
                need_pc = True
            elif source == "net":
                need_net = True
        except ValueError as e:
            print(f"Error parsing feature argument '{f_arg}': {e}")
            sys.exit(1)
            
    pc_evecs = None
    net_evecs = None
    
    if need_pc:
        print(f"Loading PC eigenvectors from {pc_evecs_path}...")
        pc_evecs = np.load(pc_evecs_path)
        print(f"PC Eigenvectors shape: {pc_evecs.shape}")
        
    if need_net:
        if not net_evecs_path.exists():
             print(f"Error: {net_evecs_path} does not exist, but 'net' features were requested.")
             sys.exit(1)
        print(f"Loading Net eigenvectors from {net_evecs_path}...")
        net_evecs = np.load(net_evecs_path)
        print(f"Net Eigenvectors shape: {net_evecs.shape}")
    
    try:
        cmap = plt.get_cmap(args.colormap)
    except ValueError:
        # Try prepending 'cmc.' for cmcrameri colormaps
        try:
            cmap = plt.get_cmap(f"cmc.{args.colormap}")
        except ValueError:
            raise ValueError(f"Colormap '{args.colormap}' is not valid.")

    direction = np.array(args.direction)
    direction = direction / np.linalg.norm(direction)
    
    all_points = []
    all_colors = []
    
    for i, (source, feat_idx) in enumerate(parsed_features):
        evecs = pc_evecs if source == "pc" else net_evecs
        
        if feat_idx >= evecs.shape[1]:
            print(f"Warning: Feature index {feat_idx} out of bounds for source '{source}' (max {evecs.shape[1]-1}). Skipping.")
            continue
            
        feat = evecs[:, feat_idx]
        norm_feat = normalize_feature(feat, symmetric=args.symmetric)
        
        # Apply colormap -> (N, 4) float RGBA
        colors_rgba = cmap(norm_feat)
        # Convert to uint8 (N, 4)
        colors_uint8 = (colors_rgba * 255).astype(np.uint8)
        
        # Offset points
        offset = direction * args.spacing * i
        shifted_points = points + offset
        
        all_points.append(shifted_points)
        all_colors.append(colors_uint8)
        
    if not all_points:
        print("No features processed. Exiting.")
        sys.exit(1)
        
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    print(f"Combined points shape: {combined_points.shape}")
    
    # Create Trimesh point cloud
    # Note: trimesh handles colors passed as 'colors' (vertex colors)
    pcd = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    pcd.export(output_path)
    print("Done.")
    
    print("\nTo render the result, run something like:")
    print(f"blender -b -P renders/render_once.py -- --input {output_path} --output {output_path.with_suffix('.png')} --radius 0.005 --resolution 1920 1080")

if __name__ == "__main__":
    main()
