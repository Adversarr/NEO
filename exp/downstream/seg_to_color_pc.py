import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path
import sys

def get_colormap(colormap_name):
    # Reference: /home/adversarr/Repo/g2pt/scripts/make_a_grid.py#L87-97
    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        try:
            import cmcrameri
            cmap = plt.get_cmap(f"cmc.{colormap_name}")
        except (ValueError, ImportError):
             print(f"Warning: Colormap '{colormap_name}' not found. Falling back to 'coolwarm'.")
             cmap = plt.get_cmap("coolwarm")
    return cmap

def main():
    parser = argparse.ArgumentParser(description="Convert segmentation results to colored point clouds.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file containing segmentation results.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for point cloud files.")
    parser.add_argument("--colormap", type=str, default="tab20", help="Colormap to use for segmentation classes.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading data from {input_path}...")
    try:
        data = torch.load(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)
    
    if not isinstance(data, list):
        print("Error: Expected a list of samples.")
        sys.exit(1)
        
    print(f"Found {len(data)} samples.")
    
    # Infer number of classes
    all_preds = [s['preds'] for s in data]
    if all_preds:
        max_cls = max([p.max().item() for p in all_preds])
        num_classes = max_cls + 1
    else:
        num_classes = 1
        
    print(f"Inferred number of classes: {num_classes}")
    
    # Get colormap
    # Using a discrete colormap based on the number of classes
    # If we pass an integer to get_cmap, it creates a discretized version
    raw_cmap = get_colormap(args.colormap)
    
    # We want distinct colors for each class. 
    # If we use the raw_cmap (continuous), we need to sample it evenly.
    colors_lut = raw_cmap(np.linspace(0, 1, num_classes))
    
    for i, sample in enumerate(data):
        points = sample['points'].numpy()
        preds = sample['preds'].numpy() # (N,) indices
        
        # Map indices to colors
        # preds contains indices [0, num_classes-1]
        # We can directly index into colors_lut
        
        # Handle case where preds might have values >= num_classes (shouldn't happen if we inferred correctly)
        # But just in case, clip or check?
        # Inference logic above guarantees preds < num_classes.
        
        sample_colors = colors_lut[preds] # (N, 4)
        
        # Convert to uint8 for trimesh/ply
        sample_colors_uint8 = (sample_colors * 255).astype(np.uint8)
        
        # Create PointCloud
        pc = trimesh.points.PointCloud(vertices=points, colors=sample_colors_uint8)
        
        out_filename = output_dir / f"sample_{i:03d}.ply"
        pc.export(out_filename)
        
    print(f"Successfully saved {len(data)} point clouds to {output_dir}")

if __name__ == "__main__":
    main()
