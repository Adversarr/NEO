"""Precompute deformation for baked MOYO dataset"""

import numpy as np
import h5py
import torch
import trimesh
import scipy.sparse as sp
import multiprocessing
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from g2pt.utils.mesh_feats import robust_mesh_laplacian as mesh_laplacian


def parse_args():
    parser = ArgumentParser(description="Convert baked MOYO .pt data to HDF5 format with sampled deformations.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the baked .pt file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output .hdf5 file.")
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of points to sample per mesh.")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers for multiprocessing.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of samples to process in one chunk to save memory.")
    parser.add_argument(
        "--h5-compression",
        type=str,
        default="lzf",
        choices=["none", "gzip", "lzf"],
        help="HDF5 compression filter for datasets",
    )
    parser.add_argument(
        "--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip"
    )
    return parser.parse_args()

def interpolate_points(mesh, tar_verts, points, face_indices):
    """
    Interpolate points on the target mesh using barycentric coordinates from the source mesh.
    """
    triangles = mesh.triangles[face_indices]
    bary = trimesh.triangles.points_to_barycentric(triangles, points)
    
    faces = mesh.faces[face_indices]
    v0 = tar_verts[faces[:, 0]]
    v1 = tar_verts[faces[:, 1]]
    v2 = tar_verts[faces[:, 2]]
    
    tar_points = (
        v0 * bary[:, 0:1] +
        v1 * bary[:, 1:2] +
        v2 * bary[:, 2:3]
    ).astype(np.float32)
    
    return tar_points

def process_single_sample(args):
    """Worker function to process a single sample."""
    i, src_verts, tar_verts, faces = args
    
    # Normalize source mesh and apply same transform to target mesh
    center = np.mean(src_verts, axis=0)
    src_verts = src_verts - center
    tar_verts = tar_verts - center
    
    max_extent = np.max(np.abs(src_verts))
    if max_extent > 0:
        src_verts = src_verts / max_extent
        tar_verts = tar_verts / max_extent
    
    # Use full vertices instead of sampling to maintain consistency with faces
    points = src_verts
    deformation = (tar_verts - src_verts).astype(np.float32)
    
    # Compute Laplacian and mass matrices
    L, M = mesh_laplacian(points, faces)
    
    # Convert stiffness matrix to COO format
    stiff_coo = sp.coo_matrix(L, copy=True)
    stiff_coo.sum_duplicates()
    
    return (
        points.astype(np.float32),
        deformation,
        stiff_coo.row.astype(np.int32),
        stiff_coo.col.astype(np.int32),
        stiff_coo.data.astype(np.float32),
        M.diagonal().astype(np.float32)
    )

def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading baked data from {input_path}")
    data = torch.load(input_path)
    
    src_verts_all = data['src_verts'].cpu().numpy()
    tar_verts_all = data['tar_verts'].cpu().numpy()
    faces = data['faces'].cpu().numpy()
    
    num_samples = src_verts_all.shape[0]
    print(f"Processing {num_samples} samples using {args.num_workers} workers in chunks of {args.chunk_size}...")

    # Pre-allocate HDF5 file and datasets
    with h5py.File(output_path, "w") as hf:
        comp = None if args.h5_compression == "none" else args.h5_compression
        comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
        
        # Create datasets
        obj_paths = [f"sample_{i:05d}" for i in range(num_samples)]
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
        
        # Determine shapes
        n_points = src_verts_all.shape[1]
        hf.create_dataset("samples", (num_samples, n_points, 3), dtype=np.float32, compression=comp, compression_opts=comp_opts)
        hf.create_dataset("deformations", (num_samples, n_points, 3), dtype=np.float32, compression=comp, compression_opts=comp_opts)
        hf.create_dataset("lumped_mass", (num_samples, n_points), dtype=np.float32, compression=comp, compression_opts=comp_opts)
        
        dt_int = h5py.special_dtype(vlen=np.dtype('int32'))
        dt_float = h5py.special_dtype(vlen=np.dtype('float32'))
        ds_row = hf.create_dataset("stiff_row", (num_samples,), dtype=dt_int)
        ds_col = hf.create_dataset("stiff_col", (num_samples,), dtype=dt_int)
        ds_val = hf.create_dataset("stiff_values", (num_samples,), dtype=dt_float)

        # Process in chunks
        for start_idx in range(0, num_samples, args.chunk_size):
            end_idx = min(start_idx + args.chunk_size, num_samples)
            print(f"Processing chunk {start_idx} to {end_idx}...")
            
            # Prepare arguments for multiprocessing for this chunk
            chunk_args = [
                (i, src_verts_all[i], tar_verts_all[i], faces) 
                for i in range(start_idx, end_idx)
            ]
            
            # Process chunk in parallel
            with multiprocessing.Pool(processes=args.num_workers) as pool:
                results = list(tqdm(pool.imap(process_single_sample, chunk_args), total=len(chunk_args), leave=False))

            # Unpack chunk results for batch writing
            # This is much faster than writing one by one, especially with compression
            chunk_len = len(results)
            chunk_samples = np.empty((chunk_len, n_points, 3), dtype=np.float32)
            chunk_deformations = np.empty((chunk_len, n_points, 3), dtype=np.float32)
            chunk_lumped_mass = np.empty((chunk_len, n_points), dtype=np.float32)
            
            # For vlen datasets, we need object arrays
            chunk_row = np.empty((chunk_len,), dtype=object)
            chunk_col = np.empty((chunk_len,), dtype=object)
            chunk_val = np.empty((chunk_len,), dtype=object)

            for i, res in enumerate(results):
                chunk_samples[i] = res[0]
                chunk_deformations[i] = res[1]
                chunk_row[i] = res[2]
                chunk_col[i] = res[3]
                chunk_val[i] = res[4]
                chunk_lumped_mass[i] = res[5]

            # Batch write to HDF5
            hf["samples"][start_idx:end_idx] = chunk_samples
            hf["deformations"][start_idx:end_idx] = chunk_deformations
            hf["lumped_mass"][start_idx:end_idx] = chunk_lumped_mass
            
            ds_row[start_idx:end_idx] = chunk_row
            ds_col[start_idx:end_idx] = chunk_col
            ds_val[start_idx:end_idx] = chunk_val

        # Optional: Store other metadata from the .pt file if present
        if 'src_poses' in data:
            hf.create_dataset("src_poses", data=data['src_poses'].cpu().numpy().astype(np.float32), compression=comp, compression_opts=comp_opts)
        if 'tar_poses' in data:
            hf.create_dataset("tar_poses", data=data['tar_poses'].cpu().numpy().astype(np.float32), compression=comp, compression_opts=comp_opts)
        if 'src_betas' in data:
            hf.create_dataset("src_betas", data=data['src_betas'].cpu().numpy().astype(np.float32), compression=comp, compression_opts=comp_opts)
        if 'tar_betas' in data:
            hf.create_dataset("tar_betas", data=data['tar_betas'].cpu().numpy().astype(np.float32), compression=comp, compression_opts=comp_opts)
        if 'genders' in data:
            hf.create_dataset("genders", data=data['genders'].cpu().numpy())
            
        # Store the faces as well
        hf.create_dataset("faces", data=faces.astype(np.int32), compression=comp, compression_opts=comp_opts)
        
        # Attributes
        hf.attrs["n_samples"] = args.n_samples
        if 'body_shape_std' in data:
            hf.attrs["body_shape_std"] = data['body_shape_std']

    print(f"Successfully written {output_path}")

if __name__ == "__main__":
    main()
