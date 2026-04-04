import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import open3d as o3d

from g2pt.utils.mesh_feats import point_cloud_laplacian

def load_mesh(path: str, target_faces: int = 30000):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    mesh.compute_vertex_normals()
    return mesh

def create_laplacian(n):
    mesh = load_mesh("data/test/bunny_mid_res.obj")
    pc = mesh.sample_points_uniformly(number_of_points=n * n)
    points = np.asarray(pc.points)
    L, M = point_cloud_laplacian(points)
    # Convert to CSR and eliminate explicit zeros to keep the sparse structure compact
    L = L.tocsr(); L.eliminate_zeros()
    M = M.tocsr(); M.eliminate_zeros()
    # Add small perturbation to keep A SPD and avoid the Neumann zero mode
    A = L + M * 1e-6
    return A, M

def make_spd_symmetric(A):
    # Enforce symmetry numerically to avoid occasional asymmetric rounding errors
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(A)
    A = (A + A.T) * 0.5
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def b_orthonormalize(X, B):
    # Orthonormalize X under the B-inner product (X^T B X = I)
    # B is sparse; S = X^T (B X) is a small (k x k) matrix
    BX = B @ X
    S = X.T @ BX
    # Build a whitening matrix via Cholesky decomposition
    try:
        R = np.linalg.cholesky(S)
        X = X @ np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # If S is nearly singular, fall back to Euclidean orthogonalization
        Q, _ = np.linalg.qr(X)
        X = Q
    return X

def solve_eigs_lobpcg(A, B, k=64, tol=1e-6, maxiter=200, seed=0, use_amg=True):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    # Initial orthogonalization under the B-inner product helps convergence and stability
    X = b_orthonormalize(X, B)

    # Preconditioner: AMG approximation of A^{-1}
    M_prec = None
    if use_amg:
        try:
            import pyamg
            t0 = time.time()
            ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric')
            M_prec = ml.aspreconditioner(cycle='V')  # Returns a LinearOperator
            t_build = time.time() - t0
            print(f"AMG preconditioner build time: {t_build:.2f} s, levels: {len(ml.levels)}")
        except Exception as e:
            print(f"AMG preconditioner unavailable or build failed; running without preconditioner: {e}")

    t1 = time.time()
    # LOBPCG: solve A x = λ B x, M is preconditioner (approximation of A^{-1})
    evals, evecs = spla.lobpcg(
        A, X, B=B, M=M_prec,
        tol=tol, maxiter=maxiter, largest=False, verbosityLevel=0
    )
    t2 = time.time()
    # Filter out non-converged pairs (lobpcg may return NaN)
    good = np.isfinite(evals)
    evals, evecs = evals[good], evecs[:, good]

    # Sort by eigenvalue
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:, idx]

    # Compute residuals
    R = A @ evecs - (B @ evecs) * evals
    # Per-column residual norm
    res = np.linalg.norm(R, axis=0) / (np.linalg.norm(A @ evecs, axis=0) + 1e-30)

    print(f"LOBPCG time: {t2 - t1:.2f} s; returned {len(evals)} eigenpairs")
    return evals, evecs, res

if __name__ == "__main__":
    # Parameters
    grid_size = 100 * 3
    num_eigenpairs = 64
    tol = 1e-6

    print("1) Generating/assembling discrete operators...")
    t0 = time.time()
    A, B = create_laplacian(grid_size)
    A = make_spd_symmetric(A)
    B = make_spd_symmetric(B)
    n = A.shape[0]
    print(f"Matrix size: {A.shape}, A.nnz={A.nnz}, B.nnz={B.nnz}, assembly time: {time.time()-t0:.2f}s")

    print("2) Solving for the k smallest eigenvalues (LOBPCG + AMG)...")
    evals, evecs, res = solve_eigs_lobpcg(A, B, k=num_eigenpairs, tol=tol, maxiter=300, seed=0, use_amg=True)

    # Print the first few results
    k_show = min(10, len(evals))
    print(f"Smallest {k_show} eigenvalues:")
    for i in range(k_show):
        print(f"  λ[{i}] = {evals[i]:.8e}, residual = {res[i]:.2e}")

    # Optional: save results
    # np.savez("eig_results.npz", evals=evals, evecs=evecs)
