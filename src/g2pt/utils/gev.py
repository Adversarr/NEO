from typing import Tuple, Literal
import torch
import numpy as np
from g2pt.utils.common import ensure_numpy
from g2pt.utils.mesh_feats import point_cloud_laplacian
from g2pt.utils.ortho_operations import qr_orthogonalization
from scipy import linalg as la
from scipy.sparse import linalg as sla
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

def dense_eigsh(L, M):
    try:
        return la.eigh(L, M)
    except Exception as e:
        print(f"Warning(eigsh): Eigenvalue decomposition failed: {e}")
        # TODO: M may be a 1D vector or sparse matrix; calling diagonal() directly may fail — use a more robust print.
        try:
            if hasattr(M, "diagonal"):
                print(f"M.diag={M.diagonal()}")
            elif isinstance(M, np.ndarray):
                if M.ndim == 1:
                    print(f"M.diag={M}")
                else:
                    print(f"M.diag={np.diagonal(M)}")
            else:
                print("M.diag=<unavailable>")
        except Exception:
            print("M.diag=<error while retrieving>")
        raise e

def dense_eigh(L, M):
    """Alias of dense_eigsh for backward naming compatibility."""
    return dense_eigsh(L, M)

def solve_gev_ground_truth(
    L: sp.csr_matrix | sp.csc_matrix | np.ndarray,
    M: sp.csr_matrix | sp.csc_matrix | np.ndarray | None,
    k: int,
    dense_threshold: int=384,
    max_iter: int=10000,
    tol: float=1e-8,
    initial_guess: np.ndarray | None = None,
    prefer='arpack'):
    """Solve the generalized eigenvalue problem (GEVP) in a given subspace.

    If the matrix is small enough, we use dense solver instead.

    If really large, use lobpcg solver. (10*dense_threshold)
    
    Args:
        L (sp.csc_matrix | np.ndarray): Stiffness/Laplacian matrix (NxN).
        M (sp.csc_matrix | np.ndarray | None): Mass matrix. Supports full (NxN), diagonal (N,), or column vector (N,1).
        k (int): Number of eigenvalues to compute.
        dense_threshold (int, optional): Threshold for using dense solver. Defaults to 2048.
        max_iter (int, optional): Maximum number of iterations for lobpcg solver. Defaults to 10000.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (eigenvalues (k,), ambient eigenvectors (N x k)).
    """
    n = L.shape[0]  # type: ignore
    k = int(min(max(k, 1), n))

    use_dense = n < dense_threshold
    use_lobpcg = n >= 30000 * dense_threshold # 3000 * 1024 = 3072000 = 3Million
    use_lobpcg = (use_lobpcg or prefer == 'lobpcg' or prefer == 'amg') and not use_dense

    if use_dense or k >= n - 1:
        Ld = L
        Md = M
        if sp.issparse(Ld):  # type: ignore[arg-type]
            Ld = Ld.toarray()  # type: ignore[attr-defined]
        if Md is not None:
            if sp.issparse(Md):  # type: ignore[arg-type]
                Md = Md.toarray()  # type: ignore[attr-defined]
            elif isinstance(Md, np.ndarray):
                if Md.ndim == 1:
                    Md = np.diag(Md)
                elif Md.ndim == 2 and Md.shape[1] == 1:
                    Md = np.diag(Md.reshape(-1))

        Ld = np.asarray(Ld, dtype=np.float64)
        if Md is not None:
            Md = np.asarray(Md, dtype=np.float64)

        if Md is None:
            evals, evecs = la.eigh(Ld)
        else:
            evals, evecs = la.eigh(Ld, Md)
        return evals[:k], evecs[:, :k]

    A = L if sp.issparse(L) else sp.csc_matrix(L)
    B = None
    Minv = None
    if M is not None:
        if isinstance(M, np.ndarray):
            if M.ndim == 1:
                B = sp.diags(M)
                Minv = sp.diags(1.0 / (M + 1e-18))
            elif M.ndim == 2 and M.shape[1] == 1:
                vec = M.reshape(-1)
                B = sp.diags(vec)
                Minv = sp.diags(1.0 / (vec + 1e-18))
            else:
                B = sp.csc_matrix(M)
        else:
            B = sp.csc_matrix(M)
            try:
                d = B.diagonal()
                if d is not None and d.size == B.shape[0]:
                    Minv = sp.diags(1.0 / (d + 1e-18))
            except Exception:
                pass

    if B is not None:
        A = A + max(1e-8, tol) * B
    else:
        A = A + max(1e-8, tol) * sp.identity(A.shape[0], format='csc')

    if use_lobpcg:
        if initial_guess is not None:
            X = initial_guess.astype(np.float64)
            if X.shape[1] > k:
                X = X[:, :k]
            elif X.shape[1] < k:
                print("No enough initial guess columns, pad with random.")
                # Pad with random if necessary, though usually k should match
                padding = np.random.randn(X.shape[0], k - X.shape[1]).astype(np.float64)
                X = np.concatenate([X, padding], axis=1)
        else:
            X = np.random.randn(A.shape[0], k).astype(np.float64)
        
        mass_vec = None
        if B is not None:
            try:
                mass_vec = B.diagonal().astype(np.float64).reshape(-1, 1)
            except Exception:
                mass_vec = None
        X = qr_orthogonalization(X, mass_vec)

        # AMG Preconditioner: use Smoothed Aggregation solver if preferred.
        M_prec = None
        if prefer == "amg":
            try:
                import pyamg

                ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
                M_prec = ml.aspreconditioner(cycle="V")
            except ImportError:
                print("Warning: pyamg not found, falling back to plain LOBPCG.")
            except Exception as e:
                print(f"Warning: Failed to build AMG preconditioner: {e}. Falling back to plain LOBPCG.")

        w, v = sla.lobpcg(A, X, B=B, M=M_prec, largest=False, tol=tol, maxiter=max_iter)
        idx = np.argsort(w)
        return w[idx][:k], v[:, idx][:, :k]

    v0 = None
    if initial_guess is not None:
        v0 = initial_guess[:, :0].astype(np.float64)

    if B is not None:
        ev, eve = eigsh(A, k=k, M=B, which="LM", sigma=0.0, v0=v0, tol=tol)
    else:
        ev, eve = eigsh(A, k=k, which="LM", sigma=0.0, v0=v0, tol=tol)

    idx = np.argsort(ev)

    # TODO: validate the error is not large.

    return ev[idx][:k], eve[:, idx][:, :k]

def solve_gev(
    L: sp.csr_matrix | sp.csc_matrix | np.ndarray | None = None,
    M: sp.csr_matrix | sp.csc_matrix | np.ndarray | None = None,
    k: int = 10,
    point_cloud: torch.Tensor | np.ndarray | None = None,
    mass: torch.Tensor | np.ndarray | None = None,
    delta: float | None = None,
    dense_threshold: int = 1024,
    max_iter: int = 10000,
    rtol: float = 1e-8,
    prefer: str = "arpack",
):
    """Unified GEV solver.

    Supports solving from (L, M) directly or from a point cloud with optional mass.
    Returns the first k eigenpairs in ascending order as numpy arrays.
    """
    if L is None or M is None:
        if point_cloud is None:
            raise ValueError("Either (L, M) or point_cloud must be provided")
        L, M, _ = create_system_for_pointcloud(
            torch.as_tensor(point_cloud) if isinstance(point_cloud, np.ndarray) else point_cloud,
            torch.as_tensor(mass) if isinstance(mass, np.ndarray) else mass,
            k,
            delta,
        )
    evals, evecs = solve_gev_ground_truth(
        L, M, k=k, dense_threshold=dense_threshold, max_iter=max_iter, tol=rtol, prefer=prefer
    )
    return evals[:k], evecs[:, :k]


def solve_gev_subspace(
    L: sp.csc_matrix | np.ndarray,
    M: sp.csc_matrix | np.ndarray | None,
    subspace_vector: np.ndarray,
):
    """
    Solve the generalized eigenvalue problem (GEVP) in a given subspace.

    Args:
        L (sp.csc_matrix | np.ndarray): Stiffness/Laplacian matrix (NxN).
        M (sp.csc_matrix | np.ndarray | None): Mass matrix. Supports full (NxN), diagonal (N,), or column vector (N,1).
        subspace_vector (np.ndarray): Subspace basis (N x D), columns assumed linearly independent.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (eigenvalues (D,), ambient eigenvectors (N x D)).

    TODO: If M is non-diagonal but shaped (N,1), the current implementation treats it as a diagonal mass; if that assumption does not hold, replace with correct sparse multiplication.
    """
    sub = subspace_vector.astype(np.float64)
    if not L.dtype == np.float64:
        L = L.astype(np.float64)
    if M is not None:
        if not M.dtype == np.float64:
            M = M.astype(np.float64)

    L_reduced = sub.T @ L @ sub  # [D, D], D is small, dense is ok
    if isinstance(M, sp.csc_matrix):
        M_reduced = sub.T @ (M @ sub)
    elif isinstance(M, np.ndarray):
        # Support 1D or (N,1) lumped mass vectors; otherwise fall back to standard matrix multiply
        if M.ndim == 1 or (M.ndim == 2 and M.shape[1] == 1):
            M_reduced = sub.T @ (M.reshape(-1, 1) * sub)
        else:
            M_reduced = sub.T @ (M @ sub)
    else:
        M_reduced = np.identity(L_reduced.shape[0], dtype=np.float64)  # Identity matrix if M is None

    eval, revec = la.eigh(L_reduced, M_reduced)
    evec = sub @ revec
    return eval, evec

def balance_stiffness(
    stiff: torch.Tensor | np.ndarray | sp.csr_matrix,
    mass: torch.Tensor | np.ndarray | sp.csr_matrix,
    delta: float,
    k: int,
) -> Tuple[torch.Tensor | np.ndarray | sp.csr_matrix, torch.Tensor | np.ndarray | sp.csr_matrix]:
    """Balance the stiffness matrix. This estimation comes from demo2d.py and vis_gev_dist.py.

    Args:
        stiff (torch.Tensor): stiffness matrix, shape (n, n).
        mass (torch.Tensor): mass matrix, shape (n, n).
        delta (float): balance factor.
        k (int): number of eigenvalues to keep.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: balanced stiffness matrix and mass matrix.
    """
    L = (stiff + stiff.T) / (2 * k ** 0.5)
    M = mass * (k ** 0.5) / mass.sum()
    return L / delta + M, M

def create_system_for_pointcloud(
    point_cloud: torch.Tensor,
    mass: torch.Tensor | None = None,
    k: int = 48,
    delta: float | None = None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    L, M = point_cloud_laplacian(ensure_numpy(point_cloud))
    if mass is not None:
        mass = ensure_numpy(mass).flatten()
        if mass.ndim == 1:
            M = sp.csr_matrix(sp.diags(mass))
        else:
            M = sp.csr_matrix(mass)

    if delta is not None:
        L, M = balance_stiffness(L, M, delta, k)
    else:
        L = L + M * 1e-8 # guarantee SPD

    w_lumped = M.diagonal().astype(np.float64)
    L = L.astype(np.float64)
    M = M.astype(np.float64)
    return L, M, w_lumped  # type: ignore


def generalize_eigh_diag_B(A: torch.Tensor, B_diag: torch.Tensor):
    """
    Solve the generalized eigenvalue problem Av = λBv, where B is a diagonal positive-definite matrix.

    Args:
        A: (..., n, n) symmetric positive-definite matrix
        B_diag: (..., n) diagonal elements of B; must all be positive

    Returns:
        eigvals: (..., n) eigenvalues in ascending order
        eigvecs: (..., n, n) B-normalized eigenvectors satisfying v^T B v = I
    """
    # 1. Compute scaling factor 1/sqrt(b_i)
    inv_sqrt_B = 1.0 / torch.sqrt(B_diag)  # (..., n)

    # 2. Form C_ij = A_ij / sqrt(b_i * b_j)
    # Using broadcasting: inv_sqrt_B[:, None] * inv_sqrt_B[None, :]
    C = A * inv_sqrt_B.unsqueeze(-1) * inv_sqrt_B.unsqueeze(-2)

    # 3. Solve the standard eigenvalue problem Cw = λw
    eigvals, w = torch.linalg.eigh(C)

    # 4. Recover eigenvectors: v_i = w_i / sqrt(b_i)
    eigvecs = w * inv_sqrt_B.unsqueeze(-1)
    
    return eigvals, eigvecs

@torch.no_grad()
def solve_gev_from_subspace_cuda(
    subspace_vector: torch.Tensor,
    point_cloud: torch.Tensor,
    mass: torch.Tensor | None = None,
    k: int = 48,
    delta: float | None = None,
    L: sp.csr_matrix | None = None,
    M: sp.csr_matrix | None = None,
    precision: Literal['32', '64'] = '64',
) -> Tuple[torch.Tensor, torch.Tensor]:
    if precision == "32":
        torch_dtype = torch.float32
        np_dtype = np.float32
    elif precision == "64":
        torch_dtype = torch.float64
        np_dtype = np.float64
    else:
        raise ValueError(f"Unsupported precision={precision!r}. Expected '32' or '64'.")

    sub = subspace_vector.to(dtype=torch_dtype)
    if L is None or M is None:
        L, M, _ = create_system_for_pointcloud(point_cloud, mass, k, delta)
    L = L.astype(np_dtype) if not L.dtype == np_dtype else L
    M = M.astype(np_dtype) if not M.dtype == np_dtype else M

    from g2pt.utils.sparse import to_torch_sparse_csr
    L_cuda = to_torch_sparse_csr(L, dtype=torch_dtype).to(subspace_vector.device)
    M_diag = M.diagonal().astype(np_dtype, copy=False)
    M_cuda = torch.from_numpy(M_diag).to(dtype=torch_dtype, device=subspace_vector.device)
    L_reduced = sub.T @ (L_cuda @ sub)  # (D, D)
    M_reduced_diag = torch.sum(sub * (M_cuda.view(-1, 1) * sub), dim=0) # (D, )
    num_sub = sub.shape[-1]
    eigval, evec = generalize_eigh_diag_B(L_reduced, M_reduced_diag)
    reconstructed_evec = (sub @ evec)[:, :num_sub]  # [N, k+1]
    return eigval, reconstructed_evec

def solve_gev_from_subspace(
    subspace_vector: torch.Tensor,
    point_cloud: torch.Tensor,
    mass: torch.Tensor | None = None,
    k: int = 48,
    delta: float | None = None,
    L: sp.csr_matrix | None = None,
    M: sp.csr_matrix | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem (GEVP) for a given subspace basis on a point cloud, and compare to dense ground truth.

    Args:
        subspace_vector (torch.Tensor): Subspace basis (N x D).
        point_cloud (torch.Tensor): Point cloud coordinates.
        mass (torch.Tensor, optional): Lumped mass (N,) or full mass (N x N). Defaults to None.
        k (int): Number of ground-truth eigenvectors to compare.
        delta (float | None): Optional balancing factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (reconstructed ambient eigenvectors from subspace (N x D), dense ground-truth eigenvectors (N x k)).
    """
    sub = ensure_numpy(subspace_vector).astype(np.float64)
    if L is None or M is None:
        L, M, _ = create_system_for_pointcloud(point_cloud, mass, k, delta)
    else:
        L = L.astype(np.float64)
        M = M.astype(np.float64)

    L_reduced = sub.T @ (L @ sub)  # [D, D], D is small, dense is ok
    M_reduced = sub.T @ (M @ sub)  # [D, D], not necessary since already normalized.

    num_sub = sub.shape[-1]
    red_eval, red_evec = dense_eigsh(L_reduced, M_reduced)
    reconstructed_evec = (sub @ red_evec)[:, :num_sub]  # [N, k+1]
    return red_eval, reconstructed_evec

def solve_gev_gt(
    point_cloud: torch.Tensor,
    mass: torch.Tensor | None = None,
    k: int = 48,
    delta: float | None = None,
    L: sp.csr_matrix | None = None,
    M: sp.csr_matrix | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem (GEVP) for a given subspace basis on a point cloud, and compare to dense ground truth.

    Args:
        point_cloud (torch.Tensor): Point cloud coordinates.
        mass (torch.Tensor, optional): Lumped mass (N,) or full mass (N x N). Defaults to None.
        k (int): Number of ground-truth eigenvectors to compare.
        delta (float | None): Optional balancing factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (reconstructed ambient eigenvectors from subspace (N x D), dense ground-truth eigenvectors (N x k)).
    """
    if L is None or M is None:
        L, M, _ = create_system_for_pointcloud(point_cloud, mass, k, delta)
    else:
        L = L.astype(np.float64)
        M = M.astype(np.float64)
    
    # ground truth
    evals, evecs = solve_gev_ground_truth(L, M, k=k)
    evecs = evecs[:, :k]  # Take only the first k eigenvectors
    return evals[:k], evecs

def solve_gev_from_subspace_with_gt(
    subspace_vector: torch.Tensor,
    point_cloud: torch.Tensor,
    mass: torch.Tensor | None = None,
    k: int = 48,
    delta: float | None = None,
    return_evals=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem (GEVP) for a given subspace basis on a point cloud, and compare to dense ground truth.

    Args:
        subspace_vector (torch.Tensor): Subspace basis (N x D).
        point_cloud (torch.Tensor): Point cloud coordinates.
        mass (torch.Tensor, optional): Lumped mass (N,) or full mass (N x N). Defaults to None.
        k (int): Number of ground-truth eigenvectors to compare.
        delta (float | None): Optional balancing factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (reconstructed ambient eigenvectors from subspace (N x D), dense ground-truth eigenvectors (N x k)).
    """
    sub = ensure_numpy(subspace_vector).astype(np.float64)
    L, M, w_lumped = create_system_for_pointcloud(point_cloud, mass, k, delta)

    L_reduced = sub.T @ L @ sub  # [D, D], D is small, dense is ok
    M_reduced = sub.T @ M @ sub  # [D, D], not necessary since already normalized.

    num_sub = sub.shape[-1]
    red_eval, red_evec = dense_eigsh(L_reduced, M_reduced)
    reconstructed_evec = (sub @ red_evec)[:, :num_sub]  # [N, k+1]

    # ground truth
    evals, evecs = solve_gev_ground_truth(L, M, k=k)
    evecs = evecs[:, :k]  # Take only the first k eigenvectors
    if return_evals:
        return reconstructed_evec, evecs, red_eval, evals
    return reconstructed_evec, evecs

def outer_cosine_similarity(
    u: np.ndarray,
    v: np.ndarray,
    M: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the outer cosine similarity between two vectors/matrices with respect to a mass matrix M.
    
    Args:
        u (np.ndarray): First vector.  [npoints, k]
        v (np.ndarray): Second vector. [npoints, l]
        M (np.ndarray | None): Mass matrix. If None, standard cosine similarity is used.
    
    Returns:
        np.ndarray: The outer cosine similarity between u and v with respect to M.
    """
    if M is not None:
        sqrt_mass = np.sqrt(M.flatten())
        u = u * sqrt_mass.reshape(-1, 1)
        v = v * sqrt_mass.reshape(-1, 1)

    utv = np.einsum('ni,nj->ij', u, v) # (k, l)
    u_norm = np.linalg.norm(u, axis=0) # (k,)
    v_norm = np.linalg.norm(v, axis=0) # (l,)
    # Guard against division by zero: add a small epsilon when any column norm is 0
    denom = np.outer(u_norm, v_norm) + 1e-12
    return utv / denom

def norm(v: np.ndarray, M=None) -> np.ndarray | float:
    if v.ndim == 1:
        if M is None:
            return np.linalg.norm(v)
        else:
            return np.sqrt(np.sum(v * (M.flatten() * v)))
    else:
        if M is None:
            return np.linalg.norm(v, axis=0)
        else:
            return np.sqrt(np.sum(v * (M.reshape(-1, 1) * v), axis=0))

def inner(u: np.ndarray, v: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the dot product of two vectors/matrices with respect to a mass matrix M.
    
    Args:
        u (np.ndarray): First vector.  [npoints, ...]
        v (np.ndarray): Second vector. [npoints, ...]
        M (np.ndarray | None): Mass matrix. If None, standard dot product is used.
    
    Returns:
        np.ndarray: The dot product of u and v with respect to M.
    """
    if M is None:
        mv = v
    else:
        if v.ndim == 1:
            mv = M.flatten() * v
        else:
            mv = M.reshape(-1, 1) * v
    return u.T @ mv  # Dot product with respect to M

def m_gram_schmidt(
    V_prev: np.ndarray,
    v: np.ndarray,
    M: np.ndarray | None = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Perform the Gram-Schmidt process with respect to a mass matrix M.
    
    Args:
        V_prev (np.ndarray): Previous orthogonal vectors.
        v (np.ndarray): The vector to orthogonalize.
        M (np.ndarray | None): Mass matrix. If None, standard Gram-Schmidt is used. [npoints]
        epsilon (float): Small value to avoid division by zero.
    """
    if M is None:
        # Standard Gram-Schmidt without mass matrix
        v_out = v - V_prev @ (V_prev.T @ v)
    else:
        v_out = v - V_prev @ (V_prev.T @ (M.flatten() * v))

    nrm_v_out = norm(v_out, M)
    return v_out / (nrm_v_out + epsilon)

def refine_eigenpairs(
    L: sp.csc_matrix,
    V_approx: np.ndarray,
    M: np.ndarray | None = None,
    max_iters: int = 10,
    rtol: float = 1e-10,
):
    n, k = V_approx.shape
    V_intermediate = np.zeros_like(V_approx)

    if not isinstance(L, sp.csc_matrix):
        L = sp.csc_matrix(L)

    if M is None:
        ident = sp.identity(n, format='csc')
    else:
        ident = sp.diags(M.flatten(), format='csc')


    # --- Step 1: Use deflated RQI to find high-precision approximate directions ---
    for i in range(k):
        v = V_approx[:, i].copy()

        for _ in range(max_iters):
            if i > 0:
                v = m_gram_schmidt(V_intermediate[:, :i], v, M)

            # Generalized Rayleigh quotient: lambda = (v^T L v) / (v^T M v)
            # TODO: If M is shaped (N,1), flatten() is used for element-wise weighting; this assumes M is a diagonal mass.
            num = np.dot(v, L @ v)
            den = np.dot(v, v) if M is None else np.dot(v, M.flatten() * v)
            lambda_rq = num / (den + 1e-18)

            try:
                A_shifted = L - lambda_rq * ident
                w = spsolve(A_shifted, v)
            except Exception as e:
                print(f"Warning: solver failed to refine eigenvector {i}: {e}. Using current vector.")
                break

            v_new = w / norm(w, M)

            # More robust convergence check: measure the angle between successive vectors
            if abs(np.dot(v, v_new)) > 1.0 - rtol:
                v = v_new
                break

            v = v_new

        V_intermediate[:, i] = v

    # --- Step 2: Rayleigh-Ritz projection and diagonalization ---
    lambdas_refined, V_refined = solve_gev_subspace(L, M, V_intermediate)

    return lambdas_refined, V_refined
