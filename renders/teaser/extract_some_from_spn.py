import argparse
import random
from pathlib import Path

import numpy as np
import trimesh

from g2pt.data.common import load_and_process_mesh
from g2pt.utils.mesh_feats import sample_points_uniformly


def parse_args():
    """
    Parse command line arguments for the extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract and sample point clouds from ShapeNet.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/ShapeNetCore.v2/",
        help="Path to the ShapeNet dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="teaser_samples",
        help="Path to the output directory to save .ply files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of meshes to sample per category.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=32768,
        help="Number of points to sample per mesh.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def normalize_pc(points):
    """
    Normalize point cloud to unit cube [-1, 1].
    
    Args:
        points (np.ndarray): (N, 3) point cloud
        
    Returns:
        np.ndarray: Normalized point cloud
    """
    # Center the point cloud
    center = np.mean(points, axis=0)
    points = points - center

    # Scale to fit in unit cube [-1, 1]
    max_extent = np.max(np.abs(points))
    if max_extent > 0:
        points = points / max_extent
        
    return points


def main():
    """
    Main execution loop: iterate through ShapeNet categories, randomly select meshes,
    sample them to high-density point clouds, normalize, and save as PLY files.
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ShapeNet category directories
    categories = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(categories)} potential categories in {input_dir}")

    total_saved = 0

    for category_dir in categories:
        category_id = category_dir.name
        
        # Find all OBJ files in this category
        # ShapeNet structure: category/model_id/models/model_normalized.obj
        all_objs = list(category_dir.glob("*/models/model_normalized.obj"))
        
        if not all_objs:
            # Try alternate structure if needed, but standard is above
            continue

        print(f"Processing category {category_id}: {len(all_objs)} objects found.")

        # Randomly select count meshes
        selected_objs = random.sample(all_objs, min(len(all_objs), args.count))

        for idx, obj_path in enumerate(selected_objs):
            try:
                # Load and process the mesh
                vertices, faces, _ = load_and_process_mesh(str(obj_path))
                
                # Sample high-density point cloud
                points = sample_points_uniformly(vertices, faces, args.n_points, seed=args.seed + idx)
                
                # Normalize point cloud
                points = normalize_pc(points)
                
                # Save as PLY file
                # Naming: CLS_idx.ply
                out_filename = f"{category_id}_{idx}.ply"
                out_path = output_dir / out_filename
                
                # Use trimesh to export point cloud
                pc = trimesh.PointCloud(vertices=points)
                pc.export(str(out_path))
                
                print(f"  Saved {out_filename}")
                total_saved += 1
                
            except Exception as e:
                print(f"  Error processing {obj_path}: {e}")

    print(f"\nSuccessfully saved {total_saved} point clouds to {output_dir}")


if __name__ == "__main__":
    main()
