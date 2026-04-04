import trimesh
import os
import glob
from argparse import ArgumentParser
import numpy as np
import fast_simplification as fs
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Remesh a folder of .obj files to a target vertex count using subdivision and QEM simplification.")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing .obj files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for remeshed files")
    parser.add_argument("--target_v", type=int, default=1024, help="Target number of vertices")
    parser.add_argument("--subdivisions", type=int, default=2, help="Number of subdivision iterations before simplification")
    return parser.parse_args()

def remesh_mesh(mesh: trimesh.Trimesh, target_v: int, subdivisions: int):
    """
    Remesh a single mesh by first subdividing it and then simplifying it to the target vertex count.
    
    Args:
        mesh: The input trimesh.Trimesh object.
        target_v: The desired number of vertices.
        subdivisions: Number of subdivision iterations to perform before simplification.
        
    Returns:
        A new trimesh.Trimesh object with approximately target_v vertices.
    """
    # 1. Subdivide to ensure we have enough vertices to simplify from
    if subdivisions > 0:
        mesh = mesh.subdivide(iterations=subdivisions)
    
    # 2. Simplify using QEM (fast_simplification)
    # In a triangle mesh, F ≈ 2V - 4. 
    # We use this heuristic to set the target face count for the simplification algorithm.
    v = mesh.vertices
    f = mesh.faces
    
    # target_count in fast_simplification refers to the number of faces.
    target_f = target_v * 2 
    
    # Perform QEM simplification
    # Note: fast_simplification is very efficient for this.
    new_v, new_f = fs.simplify(v, f, target_count=target_f)
    
    return trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)

def main():
    args = parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        print(f"Created output directory: {args.output}")
        
    # Get all .obj files in the input directory
    obj_files = sorted(glob.glob(os.path.join(args.input, "*.obj")))
    if not obj_files:
        print(f"No .obj files found in {args.input}")
        return
        
    print(f"Found {len(obj_files)} .obj files. Starting remeshing to ~{args.target_v} vertices...")
    
    # Process each file with a progress bar
    for obj_path in tqdm(obj_files):
        try:
            # Load mesh
            mesh = trimesh.load(obj_path)
            
            # Handle scenes (concatenate all geometries into one mesh)
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    print(f"Warning: {obj_path} contains an empty scene. Skipping.")
                    continue
                mesh = trimesh.util.concatenate(
                    [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            
            # Perform remeshing
            new_mesh = remesh_mesh(mesh, args.target_v, args.subdivisions)
            
            # Check if we got close to the target
            # if abs(len(new_mesh.vertices) - args.target_v) > args.target_v * 0.1:
            #     # Optional: log if the vertex count is significantly off
            #     pass
            
            # Save to output folder
            filename = os.path.basename(obj_path)
            output_path = os.path.join(args.output, filename)
            new_mesh.export(output_path)
            
        except Exception as e:
            print(f"\nError processing {obj_path}: {e}")

if __name__ == "__main__":
    main()
