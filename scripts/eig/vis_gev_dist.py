import argparse
import logging
import os
import sys
from pathlib import Path
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.mesh_feats import sample_points_uniformly
import matplotlib.pyplot as plt
import numpy as np

from g2pt.utils.mesh_feats import load_and_process_mesh, solve_gev, point_cloud_laplacian
from g2pt.utils.rot import random_rotate_3d

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Generalized Eigenvalue Distribution for 3D meshes")
    parser.add_argument(
        "--obj",
        type=str,
        default="/data/ShapeNetCore.v2/03593526/5cf4d1f50bc8b27517b73b4b5f74e5b2/models/model_normalized.obj",
        help="Path to the OBJ file to analyze",
    )
    parser.add_argument("--k", type=int, default=16, help="Number of eigenvalues to compute and plot")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="Regularization parameter for the generalized eigenvalue problem"
    )
    parser.add_argument("--output", type=str, default="gev.png", help="Output filename for the plot")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image")
    return parser.parse_args()


def validate_file_path(file_path: str) -> bool:
    """Validate that the specified file exists."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    return True


def main():
    """Main function to compute and visualize GEV distribution."""
    setup_logging()
    args = parse_arguments()

    # Validate input file
    if not validate_file_path(args.obj):
        sys.exit(1)

    try:
        # Load and process mesh
        verts, faces, mesh = load_and_process_mesh(args.obj)
        logging.info(f"Loaded mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

        n = verts.shape[0]
        # Center and normalize vertices
        verts = sample_points_uniformly(verts, faces, 1024)
        verts = verts - verts.mean(axis=0)
        verts = verts / np.abs(verts).max()

        rot = random_rotate_3d(3)
        verts = verts @ rot.T

        # Compute point cloud Laplacian
        L, M = point_cloud_laplacian(verts)


        logging.info(f"Laplacian matrix: L[0,0]={L[0, 0]:.6f}, M[0,0]={M[0, 0]:.6f}, M.sum={M.sum():.6f}")

        # # Scale matrices
        # L = L / (args.k**0.5)
        # M = M * (args.k**0.5) / M.sum()
        L, M = balance_stiffness(L, M, args.delta, args.k)

        # Solve generalized eigenvalue problem
        eval, evec = solve_gev(L + args.delta * M, M)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(eval[: args.k], "bo-", linewidth=2, markersize=4)
        plt.yscale("log")
        plt.xlabel("Eigenvalue Index", fontsize=12)
        plt.ylabel("Eigenvalue (log scale)", fontsize=12)
        plt.title(f"Generalized Eigenvalue Distribution\n{Path(args.obj).name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        logging.info(f"Plot saved as: {args.output}")

        # Show summary statistics
        logging.info(f"First {args.k} eigenvalues: {eval[: args.k]}")

    except Exception as e:
        logging.error(f"Error processing mesh: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
