from argparse import ArgumentParser
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from g2pt.utils.mesh_feats import process_single_mesh, load_and_process_mesh



def process(i, obj, args):
    try:
        # Load and process the mesh
        verts, faces, mesh = load_and_process_mesh(obj)
        output_path = Path(args.out) / f"{i:06d}"
        output_path.mkdir(parents=True, exist_ok=True)

        # convert to Open3D mesh for sampling
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        o3d_mesh.compute_vertex_normals()  # Compute vertex normals

        # Process the mesh to generate point clouds and Laplacian eigenvectors
        process_single_mesh(
            o3d_mesh,
            npoints=args.npoints,
            k=args.k,
            nsamples=args.nsamples,
            output_path=output_path
        )
        np.save(output_path / "trimesh_verts.npy", verts)
        np.save(output_path / "trimesh_faces.npy", faces)
        print(f"Processing {obj} ({i + 1}) and saving to {output_path}")

    except Exception as e:
        print(f"Error processing {obj}: {e}")


if __name__ == "__main__":
    # Load the ShapeNetCoreV2 dataset
    # The dataset is expected to be in the format compatible with the `datasets` library.
    # If you have a local copy, you can specify the path to it.

    parser = ArgumentParser(description="Generate point clouds from ShapeNet dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the ShapeNet dataset directory.")
    parser.add_argument("--out", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--npoints", type=int, default=1024, help="Number of points to sample from each mesh.")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors for Laplacian computation.")
    parser.add_argument("--nsamples", type=int, default=2, help="Number of samples of each mesh to generate.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of worker processes to use.")
    parser.add_argument('--limit', type=int, default=0, help="Limit the number of objects to process (0 for no limit).")
    parser.add_argument('--omp', type=str, default='1', help="OpenMP threads to use for parallel processing.")

    args = parser.parse_args()
    print(args)
    os.environ["OMP_NUM_THREADS"] = args.omp

    path = Path(args.path)
    out = Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    objs = list(path.glob("**/models/model_normalized.obj"))
    print(f"Found {len(objs)} objects in {path}")
    if args.limit > 0:
        objs = objs[: args.limit]
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as p:
            p.starmap(process, [(i, obj, args) for i, obj in enumerate(objs)])
    else:
        for i, obj in enumerate(objs):
            process(i, obj, args)
    print(f"Processed {len(objs)} objects in total.")
    print("Done.")
