from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import scipy.sparse.linalg as la

from g2pt.data.common import load_and_process_mesh
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = ArgumentParser(description="Benchmark PyAMG preconditioned solvers for GEV")
    parser.add_argument(
        "--obj",
        type=str,
        default="/data/ShapeNetCore.v2/03593526/5cf4d1f50bc8b27517b73b4b5f74e5b2/models/model_normalized.obj",
        help="Path to the OBJ file to analyze",
    )
    parser.add_argument("--npoints", type=int, default=1024, help="Number of points to sample from the mesh")
    parser.add_argument("--k", type=int, default=96, help="Number of eigenvalues to compute")
    parser.add_argument("--delta", type=float, default=1.0, help="Balancing parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for LOBPCG")
    parser.add_argument("--maxiter", type=int, default=80, help="Max iterations for LOBPCG")
    parser.add_argument(
        "--restart_control",
        type=int,
        default=10,
        help="Restart control parameter for LOBPCG",
    )
    parser.add_argument(
        "--amg_cycle",
        type=str,
        default="V",
        choices=["V", "W", "F"],
        help="AMG cycle type used as LOBPCG preconditioner",
    )
    parser.add_argument(
        "--amg_prec_maxiter",
        type=int,
        default=1,
        help="AMG iterations per preconditioner application (usually 1)",
    )
    parser.add_argument(
        "--amg_prec_tol",
        type=float,
        default=0.0,
        help="AMG tolerance per preconditioner application (0 disables convergence check)",
    )
    return parser.parse_args()

def _b_orthonormalize(X, B, eps=1e-12):
    BX = B @ X
    G = X.T @ BX
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, eps)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return X @ inv_sqrt


def _make_pyamg_preconditioner(ml, shape, dtype, cycle, tol, maxiter):
    """
    Wrap PyAMG multilevel solver as a LinearOperator for LOBPCG.
    
    PyAMG solve() typically handles 1D arrays (single vector).
    LOBPCG passes block vectors (N, k).
    We must loop over columns because PyAMG does not support multiple RHS natively.
    """
    def _solve(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return ml.solve(v, cycle=cycle, tol=tol, maxiter=maxiter)
        
        # Batched solve (column-wise)
        out = np.empty_like(v)
        for j in range(v.shape[1]):
            out[:, j] = ml.solve(v[:, j], cycle=cycle, tol=tol, maxiter=maxiter)
        return out

    return la.LinearOperator(shape=shape, matvec=_solve, matmat=_solve, dtype=dtype)


def main():
    args = parse_arguments()
    
    try:
        import pyamg
    except ImportError:
        print("pyamg not found. Please install it with 'pip install pyamg'")
        return

    # Load and process mesh
    vert, face, mesh = load_and_process_mesh(args.obj)
    vert = sample_points_uniformly(vert, face, args.npoints, args.seed)
    Lo, Mo = point_cloud_laplacian(vert)
    L, M = balance_stiffness(Lo, Mo, args.delta, args.k)

    print(f"Shape of L: {L.shape}, Shape of M: {M.shape}")

    # Baseline: ARPACK (Shift-Invert)
    print("\n--- ARPACK (Shift-Invert) ---")
    start = perf_counter()
    eigvals_arpack, _ = la.eigsh(L, k=args.k, M=M, sigma=0, which="LM")
    end = perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.npoints, args.k))
    X = _b_orthonormalize(X, M)

    # LOBPCG + PyAMG (Smoothed Aggregation)
    print("\n--- LOBPCG + PyAMG (Smoothed Aggregation) ---")
    start_prec = perf_counter()
    ml_sa = pyamg.smoothed_aggregation_solver(L, symmetry='symmetric')
    M_prec_sa = _make_pyamg_preconditioner(
        ml_sa, L.shape, L.dtype, args.amg_cycle, args.amg_prec_tol, args.amg_prec_maxiter
    )
    end_prec = perf_counter()
    print(f"AMG (SA) setup time: {end_prec - start_prec:.4f} seconds")

    start = perf_counter()
    eigvals_amg_sa, _ = la.lobpcg(
        L,
        X.copy(),
        B=M,
        M=M_prec_sa,
        largest=False,
        tol=args.tol,
        maxiter=args.maxiter,
        restartControl=args.restart_control,
    )
    end = perf_counter()
    print(f"Time taken (solve): {end - start:.4f} seconds")
    print(f"Total time (setup + solve): {end - start_prec:.4f} seconds")

    # LOBPCG + PyAMG (Root-Node)
    print("\n--- LOBPCG + PyAMG (Root-Node) ---")
    start_prec = perf_counter()
    ml_rn = pyamg.rootnode_solver(L, symmetry='symmetric')
    M_prec_rn = _make_pyamg_preconditioner(
        ml_rn, L.shape, L.dtype, args.amg_cycle, args.amg_prec_tol, args.amg_prec_maxiter
    )
    end_prec = perf_counter()
    print(f"AMG (RN) setup time: {end_prec - start_prec:.4f} seconds")

    start = perf_counter()
    eigvals_amg_rn, _ = la.lobpcg(
        L,
        X.copy(),
        B=M,
        M=M_prec_rn,
        largest=False,
        tol=args.tol,
        maxiter=args.maxiter,
        restartControl=args.restart_control,
    )
    end = perf_counter()
    print(f"Time taken (solve): {end - start:.4f} seconds")
    print(f"Total time (setup + solve): {end - start_prec:.4f} seconds")

    # Accuracy check (compare first 5 eigenvalues)

    print("\n--- Eigenvalue Comparison (First 5) ---")
    print(f"ARPACK:    {eigvals_arpack[:5]}")
    print(f"LOBPCG + AMG (SA):  {np.sort(eigvals_amg_sa)[:5]}")
    print(f"LOBPCG + AMG (RN):  {np.sort(eigvals_amg_rn)[:5]}")

if __name__ == '__main__':
    main()
