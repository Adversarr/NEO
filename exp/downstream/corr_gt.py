"""Extract some mesh to ply file from faust.

Expects paired files in `data_dir` (created by preprocess_corr.py):
- *_mesh.hdf5: verts(vlen float32), faces(vlen int32), vert_shapes, face_shapes
- *_corr.hdf5: evals(vlen float32), evecs(vlen float32)+evec_shapes, hks(vlen float32)+hks_shapes, corres(vlen int32)

Output:
- meshs: verts(vlen float32), faces(vlen int32). in ply format.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import trimesh

def parse_args():
    """Parse command line arguments for mesh extraction."""
    parser = ArgumentParser(description="Extract meshes from HDF5 to PLY")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save PLY files")
    parser.add_argument("--num-meshes", type=int, default=20, help="Number of meshes to extract")
    return parser.parse_args()

def main():
    """Main function to load HDF5 meshes and save them as PLY files."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all mesh hdf5 files
    mesh_files = list(data_dir.glob("*_mesh.hdf5"))
    if not mesh_files:
        print(f"No mesh HDF5 files found in {data_dir}")
        return

    for mesh_file in mesh_files:
        print(f"Processing {mesh_file.name}...")
        with h5py.File(mesh_file, "r") as hf:
            verts_vlen = hf["verts"][:]
            faces_vlen = hf["faces"][:]
            vert_shapes = hf["vert_shapes"][:]
            face_shapes = hf["face_shapes"][:]
            
            if "obj_paths" in hf:
                # obj_paths are stored as byte strings in HDF5
                obj_paths = [p.decode('utf-8') for p in hf["obj_paths"][:]]
            else:
                obj_paths = [f"mesh_{i}" for i in range(len(vert_shapes))]

            for i in range(min(len(vert_shapes), args.num_meshes)):
                # Reshape flattened vlen data back to original shapes
                v = verts_vlen[i].reshape(vert_shapes[i])
                f = faces_vlen[i].reshape(face_shapes[i])
                
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                
                # Create output filename using the original object name
                obj_name = Path(obj_paths[i]).stem
                out_path = output_dir / f"{obj_name}.obj"
                
                mesh.export(str(out_path))
                print(f"  Saved {out_path}")

if __name__ == "__main__":
    main()


