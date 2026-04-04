from argparse import ArgumentParser
import numpy as np
from robust_laplacian import point_cloud_laplacian
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
from g2pt.utils.mesh_feats import hks

if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize tetmesh's HKS with colormap.")
    parser.add_argument("mesh", type=str, help="Path to the tetmesh file")
    parser.add_argument("evecs", type=str, help="Path to the eigenvector file")
    parser.add_argument("evals", type=str, help="Path to the eigenvalues file")
    parser.add_argument("--t_min", type=float, default=1e-2, help="Minimum HKS value to visualize")
    parser.add_argument("--t_max", type=float, default=100.0, help="Maximum HKS value to visualize")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the plot")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the plot")
    parser.add_argument("--show", action='store_true', help="Show the plot instead of saving it")
    parser.add_argument("--recompute", action='store_true', help="Recompute HKS even if files exist")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    evecs_path = Path(args.evecs)
    evals_path = Path(args.evals)

    mesh = np.load(mesh_path)    # [nPoints, 3]
    evecs = np.load(evecs_path) # [nPoints, C]
    evals = np.load(evals_path) # [nPoints, C]
    # print(evals)
    evals = evals / evals.max()  # Normalize eigenvalues
    # print(np.allclose(evecs[:, 0], np.mean(evecs[:, 0])))  # Check if first eigenvector is constant
    if args.recompute:
        L, M = point_cloud_laplacian(mesh)
        evals, evecs = scipy.linalg.eigh(L.todense(), M.todense())  # Compute eigenvalues and eigenvectors

    hks = hks(evecs, evals, t_min=args.t_min, t_max=args.t_max, count=args.rows * args.cols)

    fig = plt.figure(figsize=(10, 10))
    for i in range(args.rows):
        for j in range(args.cols):
            ax = fig.add_subplot(args.rows, args.cols, i*args.cols+j+1, projection='3d')
            ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=hks[:, i*args.cols+j], cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.axis('equal')
            print(i, j, hks[:, i*args.cols+j].std())
    if args.show:
        plt.show()
    else:
        plt.savefig('tetmesh_visualization.png', dpi=300, bbox_inches='tight')
