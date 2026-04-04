#!/usr/bin/env python3
"""
Irregular mesh demo using processed ShapeNet-style data with training.

This script reads `trimesh_verts.npy` and `trimesh_faces.npy` from a dataset
directory structure similar to ProcessedShapenetAdaptor in g2pt/debug_sft.py,
then trains a subspace basis `E` and a solver `H` as in `tinyexp/demo2d.py`.

Workflow:
- Load mesh, uniformly sample a point cloud on the surface.
- Build point-cloud Laplacian `L` and lumped mass `M`.
- Define `A = L + delta * M`.
- Train `E` (Nxk) and `H` (kxk) by minimizing compliance and KKT energy terms.
- Optionally visualize and report Ritz eigenvalues within the learned subspace.

Notes:
- Loader style mirrors ProcessedShapenetAdaptor: `<base>/<id>/trimesh_verts.npy`,
  `<base>/<id>/trimesh_faces.npy`.
- We avoid compilation/make/cmake checks; only runtime numpy/scipy/torch.
- TODO: If dataset structure differs or faces are missing, adapt loader.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import trimesh
import matplotlib.pyplot as plt

from g2pt.utils.mesh_feats import sample_points_uniformly, point_cloud_laplacian
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.sparse import to_torch_sparse_csr, SymmSparseCSRMatmul

torch.set_default_dtype(torch.float32)


# ===== Data structures and loader (mirrors debug_sft.py ProcessedShapenetAdaptor) =====

@dataclass
class Geometry:
    """Pointcloud or triangle mesh geometry container."""
    verts: np.ndarray
    faces: Optional[np.ndarray] = None


class ProcessedShapenetAdaptor:
    """Loader that reads mesh data from `<base_path>/<id>/trimesh_*.npy`.

    - Expects two files per sample: `trimesh_verts.npy` and `trimesh_faces.npy`.
    - `ids` are derived from the directory names directly under `base_path`.
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        # Discover all immediate subdirectories as sample ids
        self.ids = sorted([d.name for d in self.base_path.iterdir() if d.is_dir()])
        if len(self.ids) == 0:
            print(f"Warning: no sample directories found under {self.base_path}.")

    def len(self) -> int:
        return len(self.ids)

    def get(self, idx: int) -> Geometry:
        """Load one sample's vertices and faces arrays.

        Parameters
        ----------
        idx: index into `ids` list

        Returns
        -------
        Geometry: vertices `(N,3)` and faces `(F,3)` as numpy arrays

        TODO: If faces are missing or mesh is not triangular, a fallback
        sampling method is needed; we assume triangular faces exist here.
        """
        path = self.base_path / self.ids[idx]
        verts = np.load(path / "trimesh_verts.npy")
        faces = np.load(path / "trimesh_faces.npy")
        return Geometry(verts, faces)


# ===== Utilities =====

def zero_center_and_scale(points: np.ndarray) -> np.ndarray:
    """Zero-center and scale points to fit within [-1, 1]."""
    pts = points.astype(np.float32)
    mean = np.mean(pts, axis=0, keepdims=True)
    pts = pts - mean
    scale = np.max(np.abs(pts)) + 1e-12
    return pts / scale


# ===== Subspace solver components (mirroring tinyexp/demo2d.py) =====

def m_column_norms(
    E: torch.Tensor,
    M: torch.Tensor,
    eps: float = 1e-12,
    use_sparse: bool = False,
    M_s: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute M-norms of columns and return M-normalized basis Q.

    E: `(N,k)` basis, M: `(N,N)` mass matrix (symmetric, PSD).
    Returns Q: `(N,k)` and d: `(k,)` norms (detached for stability).

    TODO: For large `N`, consider using mass diagonal to avoid dense `M` matmul.
    """
    # Use symmetric sparse CSR matmul when requested; falls back to dense.
    if use_sparse and M_s is not None:
        # NOTE: Backward uses A @ dY assuming symmetry; valid for SPD M.
        ME = SymmSparseCSRMatmul.apply(M_s, E)
    else:
        # TODO: If M_s is missing while use_sparse=True, we fall back to dense.
        ME = M @ E
    d2 = (E * ME).sum(dim=0)
    d = torch.sqrt(torch.clamp(d2, min=eps)).detach()
    Q = E / d.unsqueeze(0)
    return Q, d


def sol_apply(
    E: torch.Tensor,
    H: torch.Tensor,
    M: torch.Tensor,
    b: torch.Tensor,
    use_sparse: bool = False,
    M_s: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply subspace solver: x̂ = Q H Q^T M b, with Q M-orthonormal.

    E: `(N,k)`, H: `(k,k)`, M: `(N,N)`, b: `(B,N)` or `(N,)`.
    Returns `(x_hat, Q)`.
    """
    Q, _ = m_column_norms(E, M, use_sparse=use_sparse, M_s=M_s)
    if b.dim() == 1:
        if use_sparse and M_s is not None:
            Mb = SymmSparseCSRMatmul.apply(M_s, b)
        else:
            Mb = M @ b
        y = (Mb.unsqueeze(0) @ Q).squeeze(0)
        z = y @ H
        x = z @ Q.t()
        return x, Q
    else:
        if use_sparse and M_s is not None:
            # Compute M @ b^T, then transpose back to (B,N)
            Mb_t = SymmSparseCSRMatmul.apply(M_s, b.t())  # (N,B)
            Mb = Mb_t.t()  # (B,N)
        else:
            Mb = b @ M.t()
        y = Mb @ Q
        z = y @ H
        x = z @ Q.t()
        return x, Q


def sample_b(batch_size: int, w_lump: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sample RHS load vectors with Cov(b) ≈ M^{-1} using lumped mass.

    b_i = z_i / sqrt(w_i), where `w` are the lumped masses.
    Returns `(B,N)`.
    """
    n = w_lump.numel()
    z = torch.randn(batch_size, n, device=device)
    b = z / torch.sqrt(w_lump.to(device).unsqueeze(0))
    return b


# ===== Main demo =====
# ===== Main demo =====

def main():
    parser = argparse.ArgumentParser(description="Irregular mesh demo (train E, H on sampled point cloud)")
    parser.add_argument("--base", type=str, default="/data/processed_shapenet/", help="dataset base path")
    parser.add_argument("--idx", type=int, default=0, help="sample index to load")
    parser.add_argument("--npoints", type=int, default=2048, help="number of points to sample uniformly")
    parser.add_argument("--k", type=int, default=8, help="subspace dimension")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="mini-batch size (RHS samples)")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--delta", type=float, default=10.0, help="shift A = L + delta*M")
    parser.add_argument("--comp-weight", type=float, default=1.0, help="weight of compliance term")
    parser.add_argument("--kkt-weight", type=float, default=1.0, help="weight of KKT energy term")
    parser.add_argument("--plot-every", type=int, default=200, help="print interval")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="device")
    parser.add_argument("--savefig", type=str, default="", help="path to save figure")
    parser.add_argument("--no-show", action="store_true", help="disable interactive plot")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--scatter", action="store_true", help="plot a 3D scatter of sampled points")
    parser.add_argument("--sparse", action="store_true", help="use sparse operations for matmuls and eigensolvers")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Load geometry in the same style as debug_sft.ProcessedShapenetAdaptor
    adaptor = ProcessedShapenetAdaptor(args.base)
    if adaptor.len() == 0:
        raise FileNotFoundError(f"No samples found under {args.base}. Expect directories with trimesh_verts.npy/faces.npy.")
    idx = max(0, min(args.idx, adaptor.len() - 1))
    geom = adaptor.get(idx)
    print(f"Loaded sample id='{adaptor.ids[idx]}' with verts={geom.verts.shape}, faces={geom.faces.shape if geom.faces is not None else None}")

    # Uniform surface sampling to create an irregular point cloud
    points = geom.verts[:args.npoints, :]
    points = zero_center_and_scale(points)

    # Build point-cloud Laplacian and lumped mass
    L, M = point_cloud_laplacian(points)
    L, M = balance_stiffness(L, M, args.delta, args.k)
    mass_diag = np.asarray(M.diagonal(), dtype=np.float64)

    # Torch tensors (dense) and optional SciPy sparse
    A_np = L.todense().astype(np.float64)
    M_np = M.todense().astype(np.float64)
    # Optional sparse matrices for eigensolvers and matmuls
    A_sp = L.tocsr()
    M_sp = M.tocsr()
    N = A_np.shape[0]
    vol = float(mass_diag.sum())  # approximate domain measure

    A = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
    M_t = torch.from_numpy(M_np).to(device=device, dtype=torch.float32)
    w_lump = torch.from_numpy(mass_diag.astype(np.float32)).to(device)
    # Optional: convert to torch sparse CSR for autograd-accelerated matmuls
    A_s = to_torch_sparse_csr(A_sp).to(device=device, dtype=torch.float32) if args.sparse else None
    M_s = to_torch_sparse_csr(M_sp).to(device=device, dtype=torch.float32) if args.sparse else None

    # Trainable parameters
    E = nn.Parameter(torch.randn(N, args.k, device=device) / np.sqrt(N))
    H = nn.Parameter(torch.eye(args.k, device=device) * 0.5 + 0.01 * torch.randn(args.k, args.k, device=device))
    params = [E, H]

    optimizer = optim.Adam(params, lr=args.lr)
    gamma = 0.01 ** (1.0 / max(args.epochs, 1))  # final LR = 1e-2 * initial
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    total_hist, comp_hist, kkt_hist, lr_hist, norm_hist = [], [], [], [], []

    # Optional: compute true generalized eigenvalues for reference
    try:
        if args.sparse:
            # Use shift-invert targeting smallest generalized eigenvalues
            k_eigs = max(1, min(args.k, N - 2))
            vals, _ = spla.eigsh(A_sp, k=k_eigs, M=M_sp, sigma=0.0, which="LM")
            vals = np.sort(vals)
            true_vals = vals[:args.k]
        else:
            true_vals, _ = la.eigh(A_np, M_np, overwrite_a=True, overwrite_b=True)
            true_vals = true_vals[:args.k]
        print("True smallest generalized eigenvalues:", true_vals)
    except Exception as e:
        # TODO: If sparse/dense eig fails (e.g., ill-conditioned M or tiny N), skip and continue.
        print(f"Skipping true eigs due to: {e}")
        true_vals = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Sample RHS loads with resolution-invariant scaling
        b = sample_b(args.batch, w_lump, device)  # (B,N)

        # Subspace solution x̂
        x_hat, Q = sol_apply(E, H, M_t, b, use_sparse=args.sparse, M_s=M_s)  # (B,N)

        # Matvecs with A and M, sparse when enabled
        if args.sparse and A_s is not None and M_s is not None:
            # Ax_hat = (A @ x_hat^T)^T ; Mb = (M @ b^T)^T
            Ax_t = SymmSparseCSRMatmul.apply(A_s, x_hat.t())
            Ax_hat = Ax_t.t()
            Mb_t = SymmSparseCSRMatmul.apply(M_s, b.t())
            Mb = Mb_t.t()
        else:
            Ax_hat = x_hat @ A.t()
            Mb = b @ M_t.t()

        # Compliance: -E[(b^T M x̂)/vol]
        comp = - ((Mb * x_hat).sum(dim=1) / vol).mean()

        # Subspace energy: E[(0.5 x̂^T A x̂ - x̂^T M b)/vol]
        kkt = (0.5 * (Ax_hat * x_hat).sum(dim=1) - (Mb * x_hat).sum(dim=1)).mean() / vol

        # Asymmetric M-orthogonalization penalty (lower-triangular + left detach)
        if args.sparse and M_s is not None:
            ME = SymmSparseCSRMatmul.apply(M_s, E)
        else:
            ME = M_t @ E
        ortho = torch.tril(E.t().detach() @ ME / vol, diagonal=1)
        norm_E = (E * ME).sum(dim=0) / vol
        norm_pen = ortho.square().mean() + norm_E.mean()

        loss = args.comp_weight * comp + args.kkt_weight * kkt + 1e-8 * norm_pen
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_hist.append(loss.detach().cpu().item())
        comp_hist.append(comp.detach().cpu().item())
        kkt_hist.append(kkt.detach().cpu().item())
        lr_hist.append(optimizer.param_groups[0]["lr"])
        norm_hist.append(norm_pen.detach().cpu().item())

        if epoch % args.plot_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                E_np = E.detach().cpu().numpy()
                # Rayleigh-Ritz in the learned subspace
                K = E_np.T @ A_np @ E_np
                S = E_np.T @ M_np @ E_np
                try:
                    S_ch = np.linalg.cholesky(S)
                    S_inv = np.linalg.inv(S_ch)
                    C = S_inv @ K @ S_inv.T
                    rr_vals, U = np.linalg.eigh(C)
                    rr_vals = rr_vals[:args.k]
                except Exception as e:
                    print(f"Ritz eig failed: {e}")
                    rr_vals = None
            print(
                f"Epoch {epoch:4d} | lr {optimizer.param_groups[0]['lr']:.3e} "
                f"| total {np.mean(total_hist[-min(args.plot_every, len(total_hist)):]):.3e} "
                f"| comp {np.mean(comp_hist[-min(args.plot_every, len(comp_hist)):]):.3e} "
                f"| kkt {np.mean(kkt_hist[-min(args.plot_every, len(kkt_hist)):]):.3e} "
                f"| norm {np.mean(norm_hist[-min(args.plot_every, len(norm_hist)):]):.3e}"
            )
            if rr_vals is not None:
                print("  Ritz eigvals (learned):", rr_vals)
            if true_vals is not None:
                print("  True eigvals:          ", true_vals)

    # Optional visualization of sampled points
    if args.scatter:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap="viridis")
        ax.set_title("Sampled points (uniform on mesh surface)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        if args.savefig:
            plt.savefig(args.savefig, dpi=160)
            print(f"Saved figure to {args.savefig}")
        if not args.no_show:
            plt.show()


    # compare the eigenvectors by compute their cosine similarity
    try:
        # Recompute learned subspace generalized eigenvectors (dense subspace RR).
        E_np = E.detach().cpu().numpy()
        K = E_np.T @ A_np @ E_np
        S = E_np.T @ M_np @ E_np
        S_ch = np.linalg.cholesky(S)
        S_inv = np.linalg.inv(S_ch)
        C = S_inv @ K @ S_inv.T
        rr_vals, W = np.linalg.eigh(C)

        # Map subspace eigenvectors back to the full space: v = L^{-T} w, phi = E v
        V_sub = S_inv.T @ W  # (k,k)
        Phi = E_np @ V_sub    # (N,k)

        # True generalized eigenvectors: dense or sparse per flag
        if args.sparse:
            # TODO: For very small N or ill-conditioned M, eigsh may fail; fallback handled by try/except.
            k_eigs = max(1, min(args.k, N - 2))
            vals_sp, vecs_sp = spla.eigsh(A_sp, k=k_eigs, M=M_sp, sigma=0.0, which="LM")
            order_sp = np.argsort(vals_sp)
            vals_sp = vals_sp[order_sp]
            vecs_sp = vecs_sp[:, order_sp]
            true_vals = vals_sp[:args.k]
            TruePhi = vecs_sp[:, :min(args.k, vecs_sp.shape[1])]
        else:
            true_vals_full, true_vecs_full = la.eigh(A_np, M_np, overwrite_a=True, overwrite_b=True)
            true_vals = true_vals_full[:args.k]
            TruePhi = true_vecs_full[:, :args.k]

        # Euclidean cosine similarity matrix |<phi_i, psi_j>| / (||phi_i|| ||psi_j||)
        Phi_norms = np.linalg.norm(Phi, axis=0) + 1e-12
        True_norms = np.linalg.norm(TruePhi, axis=0) + 1e-12
        eucl_sim = np.abs((Phi.T @ TruePhi) / (Phi_norms[:, None] * True_norms[None, :]))

        # M-weighted cosine similarity matrix |<phi_i, psi_j>_M| / (||phi_i||_M ||psi_j||_M)
        cross_M = Phi.T @ M_np @ TruePhi
        Phi_mnorms = np.sqrt(np.clip(np.diag(Phi.T @ M_np @ Phi), a_min=1e-12, a_max=None))
        True_mnorms = np.sqrt(np.clip(np.diag(TruePhi.T @ M_np @ TruePhi), a_min=1e-12, a_max=None))
        m_sim = np.abs(cross_M / (Phi_mnorms[:, None] * True_mnorms[None, :]))

        # Report best matching true mode for each learned mode using M-weighted similarity
        best_j = np.argmax(m_sim, axis=1)
        diag_sim = m_sim[np.arange(args.k), best_j]

        np.set_printoptions(precision=3, suppress=True)
        print("\nEigenvector comparison:")
        print("  Learned Ritz eigenvalues:", rr_vals[:args.k])
        print("  True eigenvalues:", true_vals)
        print("  Cosine similarity (Euclidean) [learned x true]:")
        print(eucl_sim)
        print("  Cosine similarity (M-weighted) [learned x true]:")
        print(m_sim)
        print("  Best match index per learned mode (M-weighted):", best_j)
        print("  Best-match M-cosine per learned mode:", diag_sim)
        # Also report Euclidean best matches per learned mode
        best_j_eucl = np.argmax(eucl_sim, axis=1)
        diag_eucl = eucl_sim[np.arange(args.k), best_j_eucl]
        print("  Best match index per learned mode (Euclidean):", best_j_eucl)
        print("  Best-match cosine per learned mode (Euclidean):", diag_eucl)
    except Exception as e:
        # TODO: Cosine similarity skipped if eig computation fails (e.g., S not SPD or la.eigh fails).
        # Reason: Cholesky and dense generalized eig require well-conditioned SPD matrices.
        print(f"Skipping eigenvector comparison due to: {e}")

if __name__ == "__main__":
    main()