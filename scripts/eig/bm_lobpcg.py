from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from g2pt.data.common import load_and_process_mesh
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = ArgumentParser(description="Visualize Generalized Eigenvalue Distribution for 3D meshes")
    parser.add_argument(
        "--obj",
        type=str,
        default="/data/ShapeNetCore.v2/03593526/5cf4d1f50bc8b27517b73b4b5f74e5b2/models/model_normalized.obj",
        help="Path to the OBJ file to analyze",
    )
    parser.add_argument("--npoints", type=int, default=1024, help="Number of points to sample from the mesh")
    parser.add_argument("--k", type=int, default=96, help="Number of eigenvalues to compute and plot")
    parser.add_argument("--delta", type=float, default=1.0, help="Balancing parameter")
    parser.add_argument("--output", type=str, default="gev.png", help="Output filename for the plot")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_arguments()
    vert, face, mesh = load_and_process_mesh(args.obj)
    vert = sample_points_uniformly(vert, face, args.npoints, args.seed)
    Lo, Mo = point_cloud_laplacian(vert)
    L, M = balance_stiffness(Lo, Mo, args.delta, args.k)
    # L, M = Lo, Mo

    print(f"Shape of L: {L.shape}, Shape of M: {M.shape}")


    start = perf_counter()
    eigvals1, eigvecs1 = la.eigsh(L, k=args.k, M=M, sigma=0, which="LM")
    end = perf_counter()
    print(f"Time taken for eigsh (shifted): {end - start:.4f} seconds")

    start = perf_counter()
    eigvals2, eigvecs2 = la.eigsh(L, k=args.k, M=M, which="SM")
    end = perf_counter()
    print(f"Time taken for eigsh (SM): {end - start:.4f} seconds")

    start = perf_counter()
    eigvals3, eigvecs3 = la.lobpcg(L, np.random.rand(args.npoints, args.k), B=M, maxiter=args.npoints, largest=False)
    end = perf_counter()
    print(f"Time taken for lobpcg: {end - start:.4f} seconds")

    if args.npoints < 1024:
        print("Compute dense ground truth evals")
        start = perf_counter()
        eigvals4, eigvecs4 = la.eigsh(L.todense(), k=args.k, M=M.todense(), which="SM")
        end = perf_counter()
        print(f"Time taken for eigsh (dense): {end - start:.4f} seconds")

        print("Check eigvals agreement")
        print(f"Close={np.allclose(eigvals1, eigvals4)}, distance={np.linalg.norm(eigvals1 - eigvals4)}")
        print(f"Close={np.allclose(eigvals2, eigvals4)}, distance={np.linalg.norm(eigvals2 - eigvals4)}")
        print(f"Close={np.allclose(eigvals3, eigvals4)}, distance={np.linalg.norm(eigvals3 - eigvals4)}")

if __name__ == '__main__':
    main()