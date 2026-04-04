from fast_robust_laplacian import point_cloud_laplacian as fast_pcl
from fast_robust_laplacian import set_print_timing, point_cloud_laplacian_batched
import argparse
import numpy as np
from scipy import sparse as sp
from time import perf_counter
from robust_laplacian import point_cloud_laplacian as pcl

from g2pt.utils.mesh_feats import load_and_process_mesh, sample_points_uniformly

import torch
def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Generalized Eigenvalue Distribution for 3D meshes")
    parser.add_argument(
        "--obj",
        type=str,
        default="/data/ShapeNetCore.v2/03593526/5cf4d1f50bc8b27517b73b4b5f74e5b2/models/model_normalized.obj",
        help="Path to the OBJ file to analyze",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=2**16,
        help="Number of points to sample from the mesh",
    )
    parser.add_argument(
        "--print_timing",
        action="store_true",
        help="Print timing information for each step",
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_print_timing(args.print_timing)

    # load mesh
    verts, faces, mesh = load_and_process_mesh(args.obj)
    n_points = int(2 ** 16)
    verts = sample_points_uniformly(verts, faces, n_points)
    print(f'#points={n_points}')

    # compute laplacian
    print("computing fast laplacian...")
    start = perf_counter()
    L_fast, M_fast = fast_pcl(verts)
    print(f"fast laplacian computed in {perf_counter() - start:.4f} seconds")
    n = 8
    start = perf_counter()
    bL , bM = point_cloud_laplacian_batched([verts] * n)
    dt = perf_counter() - start
    print(f"fast laplacian (batched) computed in {dt:.4f} seconds, {dt / n:.4f} seconds per point")

    print("computing robust laplacian...")
    start = perf_counter()
    L, M = pcl(verts)
    print(f"original robust laplacian computed in {perf_counter() - start:.4f} seconds")

    # compare the two laplacians
    diff_L = sp.csr_matrix(L_fast - L).data
    diff_M = sp.csr_matrix(M_fast - M).data
    relative_err_L = np.linalg.norm(diff_L) / np.linalg.norm(L_fast.data)
    relative_err_M = np.linalg.norm(diff_M) / np.linalg.norm(M_fast.data)

    print("|dL|=", np.linalg.norm(diff_L))
    print("|dM|=", np.linalg.norm(diff_M))
    print("relative_err_L=", relative_err_L)
    print("relative_err_M=", relative_err_M)

if __name__ == "__main__":
    main()