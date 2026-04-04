#!/usr/bin/env python3
"""
Lightning version of irregular point cloud subspace solver demonstration.

This script independently implements the following workflow without referencing other implementations in the repository:
- Load `trimesh_verts.npy` and `trimesh_faces.npy`;
- Sample point cloud uniformly on mesh surface based on triangle face areas;
- Build kNN graph Laplacian matrix `L` and lumped mass matrix `M` from point cloud;
- Define `A = L + delta * M`;
- Use Lightning to train subspace basis `E (N×k)` and solver `H (k×k)`, optimizing compliance and KKT energy;
- Optional: Periodically compute Rayleigh-Ritz subspace eigenvalue estimates during training, and report ground truth comparison and eigenvector cosine similarity at training end.

Dependencies: numpy, scipy, torch, lightning (or pytorch_lightning), matplotlib (visualization only).

TODO: If faces are missing or data structure differs from assumptions, provide fallback sampling strategy and more robust adjacency construction; reason: current sampling uses triangular faces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import scipy.linalg as la

from g2pt.utils.ortho_operations import qr_orthogonalization
from g2pt.utils.mesh_feats import point_cloud_laplacian
from g2pt.utils.gev import balance_stiffness, solve_gev_from_subspace_with_gt
from g2pt.utils.common import ensure_numpy

# Lightning compatibility imports (prefer lightning>=2.x, fall back to pytorch_lightning)
try:
    from lightning.pytorch import LightningModule, Trainer
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    try:
        import pytorch_lightning as pl  # type: ignore
        LightningModule = pl.LightningModule  # type: ignore
        Trainer = pl.Trainer  # type: ignore
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Lightning not installed. Please install 'lightning' or 'pytorch-lightning'."
        ) from e

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # Only used for visualization


torch.set_default_dtype(torch.float32)


# ===== Data structures and loading =====

@dataclass
class Geometry:
    """Container: triangular mesh or point cloud geometry.

    verts: (N,3) vertex coordinates; faces: (F,3) triangle face indices (optional).
    """

    verts: np.ndarray
    faces: Optional[np.ndarray] = None


def discover_ids(base_path: str) -> list[str]:
    """Enumerate all sample IDs in the dataset (subdirectory names as IDs)."""
    base = Path(base_path)
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def load_mesh_npy(base_path: str, idx: int) -> Tuple[Geometry, str]:
    """Load a sample from `<base>/<id>/trimesh_verts.npy` and `trimesh_faces.npy`.

    Returns (Geometry, id). Raises exception if faces are missing.
    TODO: If faces are missing, provide point cloud approximation sampling; reason: current sampling depends on face areas.
    """
    ids = discover_ids(base_path)
    if len(ids) == 0:
        raise FileNotFoundError(f"No sample directories found under {base_path}")
    idx = max(0, min(idx, len(ids) - 1))
    sid = ids[idx]
    path = Path(base_path) / sid
    verts = np.load(path / "trimesh_verts.npy")
    faces_path = path / "trimesh_faces.npy"
    if not faces_path.exists():
        raise FileNotFoundError(f"Missing faces: {faces_path}")
    faces = np.load(faces_path)
    return Geometry(verts=verts, faces=faces), sid


# ===== Point cloud sampling and operator construction =====

def sample_points_on_mesh(verts: np.ndarray, faces: np.ndarray, npoints: int, seed: int = 0) -> np.ndarray:
    """Sample point cloud uniformly on mesh surface based on triangle face areas.

    Uses standard uniform triangle sampling:
    - Compute area of each triangle;
    - Randomly select faces by area distribution;
    - Generate (u, v) and construct barycentric coordinates (with sqrt transform for uniformity).

    Returns (npoints, 3).
    """
    rng = np.random.default_rng(seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if np.sum(areas) <= 0:
        raise ValueError("Total face area is 0, cannot perform uniform sampling")
    prob = areas / np.sum(areas)
    # Select face indices
    tri_idx = rng.choice(len(faces), size=npoints, p=prob)
    a = verts[faces[tri_idx, 0]]
    b = verts[faces[tri_idx, 1]]
    c = verts[faces[tri_idx, 2]]
    r1 = rng.random(npoints)
    r2 = rng.random(npoints)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    pts = (u[:, None] * a) + (v[:, None] * b) + (w[:, None] * c)
    return pts.astype(np.float32)


def zero_center_and_scale(points: np.ndarray) -> np.ndarray:
    """Zero-mean and scale to approximately [-1, 1] range."""
    pts = points.astype(np.float32)
    mean = np.mean(pts, axis=0, keepdims=True)
    pts = pts - mean
    scale = np.max(np.abs(pts)) + 1e-12
    return pts / scale


# ===== Subspace LightningModule =====

class _EpochCounterDataset(Dataset):
    """Placeholder dataset to drive Lightning training loop.

    Each sample is just a placeholder; training step samples `b` independently.
    """

    def __init__(self, length: int = 1) -> None:
        self.length = max(1, int(length))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return 0


class _FixedSampleDataset(Dataset):
    """
    Fixed-sample dataset: always returns the same sample, with a structure similar to
    `__getitem__` in `g2pt/debug_sft.py`.

    - points: torch f32 dense [N, 3]
    - stiffness: scipy.sparse coo_matrix [N, N]
    - lumped_mass: torch f32 dense [N, 1]

    TODO: If batching and sparse-matrix tensorization are needed in the future,
    provide a custom `collate_fn`; reason: `scipy.sparse` cannot be merged into
    batched tensors by `DataLoader` by default.
    """

    def __init__(self, points: np.ndarray, A_np: np.ndarray, M_np: np.ndarray, length: int = 1) -> None:
        self.length = max(1, int(length))

        # Pre-build and cache tensors/matrices to avoid repeated conversion in `__getitem__`
        pts = points.astype(np.float32)
        self.points_tensor = torch.from_numpy(pts)

        # `stiffness` uses the balanced stiffness matrix (corresponding to `L_b` in `debug_sft`)
        self.stiffness_coo = sp.coo_matrix(np.asarray(A_np, dtype=np.float32), copy=True)

        # `lumped_mass` is the diagonal of the balanced mass matrix (corresponding to
        # `M_b.diagonal()` in `debug_sft`)
        mass_diag = np.diag(np.asarray(M_np, dtype=np.float32)).astype(np.float32)
        self.mass_tensor = torch.from_numpy(mass_diag).unsqueeze(-1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        # Always return the same sample and ignore `idx` (equivalent to fixing the index at 0)
        return {
            "points": self.points_tensor.clone(),
            "stiffness": self.stiffness_coo.copy(),
            "lumped_mass": self.mass_tensor.clone(),
        }


def _collate_fixed_sample(batch) -> dict:
    """
    Collate function: similar in design to `collate_fn` in `g2pt/sft.py`.

    - Input: `list[dict]`, where each dict contains `'points'`, `'stiffness'`
      (`scipy.sparse.coo`), and `'lumped_mass'`.
    - Output: instead of returning `scipy.sparse` directly, returns the sparse
      indices and values of the block-diagonal merge, together with stacked
      `points` and `lumped_mass`, to simplify downstream tensorization.

    TODO: If sparse tensors need to be used directly on GPU, convert
    `indices`/`values` to `torch.sparse_coo_tensor` here; reason: the current
    sparse indices + values are handled by the upper layer.
    """
    # Stack dense tensors
    points = torch.stack([item["points"] for item in batch], dim=0)  # (B, N, 3)
    lumped_mass = torch.stack([item["lumped_mass"] for item in batch], dim=0)  # (B, N, 1)

    # Merge scipy.sparse coo into a block-diagonal larger sparse matrix
    rows = []
    cols = []
    values = []
    npoints = points.shape[1]
    for i, item in enumerate(batch):
        coo: sp.coo_matrix = item["stiffness"]
        rows.append(coo.row + i * npoints)
        cols.append(coo.col + i * npoints)
        values.append(coo.data)

    larger_matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(npoints * len(batch), npoints * len(batch)),
        copy=True,
    )
    larger_matrix.sum_duplicates()

    stiff_indices = np.vstack([larger_matrix.row, larger_matrix.col])
    stiff_values = larger_matrix.data

    return {
        "points": points.to(dtype=torch.float32),
        "stiff_indices": torch.from_numpy(stiff_indices).to(torch.long),
        "stiff_values": torch.from_numpy(stiff_values).to(torch.float32),
        "lumped_mass": lumped_mass.to(dtype=torch.float32),
    }


class SubspaceSolverLit(LightningModule):
    """Lightning module: trains the subspace basis `E` and subspace solver `H`.

    The objective includes compliance and KKT energy, with an additional penalty
    for the M-orthogonality constraint.
    """

    def __init__(
        self,
        A_np: np.ndarray,
        M_np: np.ndarray,
        points: np.ndarray,
        k: int = 8,
        batch_size: int = 32,
        lr: float = 1e-2,
        max_epochs: int = 1000,
        comp_weight: float = 1.0,
        kkt_weight: float = 1.0,
        grad_clip_norm: float = 1.0,
        plot_every: int = 200,
        steps_per_epoch: int = 200,
        compute_true_eigs: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["A_np", "M_np"])  # Avoid writing huge matrices into `hparams`

        N = int(A_np.shape[0])
        assert A_np.shape == M_np.shape == (N, N), "A and M must have the same shape"

        # Constant buffers (not trainable)
        self.register_buffer("A", torch.as_tensor(A_np, dtype=torch.float32))
        self.register_buffer("M", torch.as_tensor(M_np, dtype=torch.float32))

        # Lumped mass and volume (domain-measure approximation)
        w_lump_np = np.array(np.diag(M_np), dtype=np.float32)
        self.w_lump_np = w_lump_np
        self.register_buffer("w_lump", torch.from_numpy(w_lump_np))
        self.vol = float(np.sum(w_lump_np))

        # Trainable parameters
        self.E = nn.Parameter(torch.randn(N, k))
        # Set the subspace solver `H` to shape `(k x k)`
        # self.H = nn.Parameter(torch.eye(k) * 0.5 + 0.01 * torch.randn(k, k))
        self.H = nn.Parameter(torch.randn(N, k))
        self.scale = nn.Parameter(torch.randn(1))
        self.points = points

        # Bookkeeping/control parameters
        self.N = N
        self.k = k
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.comp_weight = comp_weight
        self.kkt_weight = kkt_weight
        self.grad_clip_norm = grad_clip_norm
        self.plot_every = plot_every
        self.steps_per_epoch = steps_per_epoch
        self.compute_true_eigs = compute_true_eigs

        # Training diagnostics
        self.last_rr_vals: Optional[np.ndarray] = None
        self.true_vals: Optional[np.ndarray] = None

    # ===== Linear algebra and sampling utilities =====
    def m_column_norms(self, E: torch.Tensor, M: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the M-norm of each column vector and return the M-normalized basis `Q` and norms `d`."""
        ME = M * E
        d2 = (E * ME).sum(dim=1) # (b, k)
        d = torch.sqrt(torch.clamp(d2, min=eps)).detach()
        Q = E / d.unsqueeze(1) # (b, N, k)
        return Q, d

    def sol_apply(self, E: torch.Tensor, H: torch.Tensor, M: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the subspace solve: `x_hat = Q H Q^T M b`, where `Q` is the M-orthogonalized basis."""
        # b: (b, n, s)
        b = b.permute(0, 2, 1)  # (b, s, n)
        Q, _ = self.m_column_norms(E, M)
        # Mb = torch.bmm(b, M)  # (B,s,n)
        Mb = M.mT * b
        x = torch.nn.functional.softplus(self.scale) * torch.bmm(torch.bmm(Mb, H), Q.mT)
        x = x.permute(0, 2, 1)  # (b, s, n)
        return x, Q

    def sample_b(self, batch_size: int) -> torch.Tensor:
        """Sample RHS load vectors so that `Cov(b) ~= M^{-1}` using lumped mass."""
        n = self.w_lump.numel()
        z = torch.randn(batch_size, n, device=self.w_lump.device, dtype=self.w_lump.dtype)
        b = z / torch.sqrt(self.w_lump.unsqueeze(0))
        return b

    # ===== Lightning hooks =====
    def training_step(self, batch, batch_idx: int):
        # Use data from the batch (`stiff_indices`/`stiff_values`/`lumped_mass`)
        # Note: assume each sample has the same point count `N` as the model, and
        # since the dataset is fixed-sample, the batch contains the same geometry.
        B = int(batch["points"].shape[0])
        Np = int(batch["points"].shape[1])
        assert Np == self.N, "Batch points count must match model N"

        # Load sampling: scale by each sample's diagonal `lumped_mass` so that `Cov(b) ~= M^{-1}`
        s = self.batch_size
        w_lump_b = batch["lumped_mass"].to(device=self.w_lump.device, dtype=self.w_lump.dtype)  # (B,N,1)
        z = torch.randn(B, self.N, s, device=self.w_lump.device, dtype=self.w_lump.dtype)
        b = z / torch.sqrt(torch.clamp(w_lump_b, min=1e-12))

        # Subspace solve (still using the module's `M`/`E`/`H`)
        M = w_lump_b
        x_hat, Q = self.sol_apply(self.E.unsqueeze(0), self.H.unsqueeze(0), M, b)  # (B,N,s)

        # Use the batch-provided stiffness (block-diagonal COO indices + values) to compute `A x_hat`
        stiff_idx = batch["stiff_indices"].to(device=self.A.device)
        stiff_val = batch["stiff_values"].to(device=self.A.device, dtype=self.A.dtype)
        A_bd = torch.sparse_coo_tensor(stiff_idx, stiff_val, size=(B * Np, B * Np), device=self.A.device, dtype=self.A.dtype).coalesce()
        x_flat = x_hat.reshape(B * Np, s).to(device=self.A.device, dtype=self.A.dtype)
        Ax_flat = torch.sparse.mm(A_bd, x_flat)
        Ax_hat = Ax_flat.reshape(B, Np, s)

        # Use the batch `lumped_mass` (diagonal) to compute `M b`
        Mb = b * w_lump_b  # (B,N,s)
        energy = (Ax_hat * x_hat).sum(dim=1)
        load = (Mb * x_hat).sum(dim=1)
        sigma = load / energy
        x_hat = sigma * x_hat
        Ax_hat = sigma * Ax_hat

        vol_b = torch.clamp(w_lump_b.sum(dim=1, keepdim=True), min=1e-12) # (B,1,1)
        comp = - ((Mb * x_hat).sum(dim=1) / vol_b).mean()
        kkt = (0.5 * (Ax_hat * x_hat).sum(dim=1) - (Mb * x_hat).sum(dim=1)).mean() / vol_b.mean()

        # ME = M @ self.E # (B,N,k)
        ME = (M * self.E.unsqueeze(0)).squeeze(0)
        ortho = torch.tril(self.E.T.detach() @ ME / self.vol, diagonal=1)
        norm_E = (self.E * ME).sum(dim=0) / self.vol
        norm_pen = ortho.square().mean() + norm_E.mean()

        loss = self.comp_weight * comp + self.kkt_weight * kkt + 1e-8 * norm_pen

        # Log metrics
        # Pass `batch_size=B` explicitly to avoid Lightning iterating into
        # `scipy.sparse` and raising a `TypeError` while inferring batch size.
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("comp", comp, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("kkt", kkt, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("norm", norm_pen, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log("scale", torch.nn.functional.softplus(self.scale), on_step=True, prog_bar=True, batch_size=B)
        self.log("lr", torch.tensor(self.optimizers().param_groups[0]["lr"]) if self.optimizers() else torch.tensor(self.lr), on_step=True, on_epoch=False, prog_bar=False, batch_size=B)

        return loss

    def on_train_epoch_end(self) -> None:
        # Periodically estimate subspace Ritz eigenvalues
        epoch = int(self.current_epoch)
        if (epoch + 1) % max(1, self.plot_every) == 0 or epoch == 0 or (epoch + 1) == self.max_epochs:
            with torch.no_grad():
                E_np = self.E.detach().cpu().numpy()
                A_np = self.A.detach().cpu().numpy()
                M_np = self.M.detach().cpu().numpy()
                K = E_np.T @ A_np @ E_np
                S = E_np.T @ M_np @ E_np
                try:
                    S_ch = np.linalg.cholesky(S)
                    S_inv = np.linalg.inv(S_ch)
                    C = S_inv @ K @ S_inv.T
                    rr_vals, rr_evecs = np.linalg.eigh(C)
                    rr_vals = rr_vals[: self.k]
                    rr_evecs = rr_evecs[:, : self.k]
                    self.last_rr_vals = rr_vals

                    subspace_vector = qr_orthogonalization(E_np, np.diag(M_np).reshape(-1, 1))
                    rec, gt = solve_gev_from_subspace_with_gt(
                        subspace_vector,
                        point_cloud=self.points,
                        mass=self.w_lump_np,
                        k=self.k,
                        delta=1,
                    )
                    dot_rec_gt = np.einsum("ni,nj,n->ij", rec, gt, self.w_lump_np)
                    print(1 - np.max(np.abs(dot_rec_gt), axis=1))
                except Exception as e:
                    # TODO: Ritz estimation fails if `S` is not SPD or is ill-conditioned;
                    # Cholesky requires SPD.
                    self.print(f"Ritz eigenvalue estimation failed: {e}")
                    self.last_rr_vals = None

            if self.last_rr_vals is not None:
                self.print(f"Epoch {epoch+1} | Ritz eigvals: {np.array2string(self.last_rr_vals, precision=5, suppress_small=True)}")

    def on_train_epoch_start(self) -> None:
        # Optional: compute ground-truth generalized eigenvalues (dense only, for smaller problems)
        if not self.compute_true_eigs or self.current_epoch > 0:
            return
        try:
            N = self.N
            # Apply a simple size limit to avoid excessive runtime
            if N <= 4096:
                A_np = self.A.detach().cpu().numpy()
                M_np = self.M.detach().cpu().numpy()
                vals, _ = la.eigh(A_np, M_np, overwrite_a=False, overwrite_b=False)
                vals_k = vals[: self.k]
                self.true_vals = vals_k
                self.print(f"True eigvals: {np.array2string(vals_k, precision=5, suppress_small=True)}")
            else:
                # TODO: For large-scale problems, use a sparse generalized eigensolver
                # (for example, `eigsh` + shift-invert); the dense method is too slow at `O(N^3)`.
                print(f"Skipping ground-truth eigenvalue computation for large problem size {N}")
                self.true_vals = None
        except Exception as e:
            self.print(f"Skipping ground-truth eigenvalue computation: {e}")
            self.true_vals = None

    def on_train_end(self) -> None:
        # After training, try comparing eigenvectors (Euclidean and M-weighted cosine similarity)
        try:
            E_np = self.E.detach().cpu().numpy()
            A_np = self.A.detach().cpu().numpy()
            M_np = self.M.detach().cpu().numpy()
            K = E_np.T @ A_np @ E_np
            S = E_np.T @ M_np @ E_np
            S_ch = np.linalg.cholesky(S)
            S_inv = np.linalg.inv(S_ch)
            C = S_inv @ K @ S_inv.T
            rr_vals, W = np.linalg.eigh(C)
            V_sub = S_inv.T @ W
            Phi = E_np @ V_sub

            if self.true_vals is None:
                # If ground truth is unavailable, run a dense generalized eigendecomposition (may be slow)
                vals_full, vecs_full = la.eigh(A_np, M_np, overwrite_a=False, overwrite_b=False)
                true_vals = vals_full[: self.k]
                TruePhi = vecs_full[:, : self.k]
            else:
                true_vals = self.true_vals
                # For simplicity, reuse the full dense eigenvectors here (or reuse
                # them if already computed in `on_fit_start`)
                vals_full, vecs_full = la.eigh(A_np, M_np, overwrite_a=False, overwrite_b=False)
                TruePhi = vecs_full[:, : self.k]

            # Cosine similarity
            Phi_norms = np.linalg.norm(Phi, axis=0) + 1e-12
            True_norms = np.linalg.norm(TruePhi, axis=0) + 1e-12
            eucl_sim = np.abs((Phi.T @ TruePhi) / (Phi_norms[:, None] * True_norms[None, :]))

            cross_M = Phi.T @ M_np @ TruePhi
            Phi_mnorms = np.sqrt(np.clip(np.diag(Phi.T @ M_np @ Phi), a_min=1e-12, a_max=None))
            True_mnorms = np.sqrt(np.clip(np.diag(TruePhi.T @ M_np @ TruePhi), a_min=1e-12, a_max=None))
            m_sim = np.abs(cross_M / (Phi_mnorms[:, None] * True_mnorms[None, :]))

            best_j = np.argmax(m_sim, axis=1)
            diag_sim = m_sim[np.arange(self.k), best_j]
            best_j_eucl = np.argmax(eucl_sim, axis=1)
            diag_eucl = eucl_sim[np.arange(self.k), best_j_eucl]

            np.set_printoptions(precision=3, suppress=True)
            print("\nEigenvector comparison:")
            print(f"  Learned Ritz eigenvalues: {rr_vals[: self.k]}")
            print(f"  True eigenvalues: {true_vals}")
            print("  Cosine similarity (Euclidean) [learned x true]:")
            print(eucl_sim)
            print("  Cosine similarity (M-weighted) [learned x true]:")
            print(m_sim)
            print(f"  Best match index per learned mode (M-weighted): {best_j}")
            print(f"  Best-match M-cosine per learned mode: {diag_sim}")
            print(f"  Best match index per learned mode (Euclidean): {best_j_eucl}")
            print(f"  Best-match cosine per learned mode (Euclidean): {diag_eucl}")
        except Exception as e:
            # TODO: If generalized eigendecomposition fails (for example, if `S`
            # is not SPD), skip similarity reporting for numerical stability.
            self.print(f"Skipping eigenvector comparison: {e}")

    def configure_optimizers(self):
        opt = optim.Adam([self.E, self.H, self.scale], lr=self.lr)
        gamma = 0.01 ** (1.0 / max(self.max_epochs, 1))
        sch = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # Minimal `train_dataloader`: used only to drive the training loop
    def train_dataloader(self):
        # Use a fixed-sample dataset whose structure matches
        # `debug_sft.__getitem__`; fix the index at 0.
        # Since the training step does not consume batch contents and the loader
        # only drives the loop, set `length = steps_per_epoch`.
        A_np = self.A.detach().cpu().numpy()
        M_np = self.M.detach().cpu().numpy()
        ds = _FixedSampleDataset(self.points, A_np, M_np, length=self.steps_per_epoch)
        # Use an SFT-style `collate_fn` to merge `scipy.sparse` into indices +
        # values, avoiding the sparse-iteration issue in the default collate.
        return DataLoader(ds, batch_size=1, collate_fn=_collate_fixed_sample)


# ===== Main entry =====

def main():
    parser = argparse.ArgumentParser(description="Lightning demo for irregular point-cloud subspace training")
    parser.add_argument("--base", type=str, default="/data/processed_shapenet/", help="Dataset root directory")
    parser.add_argument("--idx", type=int, default=0, help="Sample index to load")
    parser.add_argument("--npoints", type=int, default=512, help="Number of sampled points")
    parser.add_argument("--k", type=int, default=12, help="Subspace dimension")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Number of RHS samples per step")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--delta", type=float, default=1.0, help="Balance term")
    parser.add_argument("--comp-weight", type=float, default=1.0, help="Compliance weight")
    parser.add_argument("--kkt-weight", type=float, default=1.0, help="KKT energy weight")
    parser.add_argument("--plot-every", type=int, default=200, help="Ritz print interval (epochs)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument("--scatter", action="store_true", help="Plot the point-cloud scatter after training")
    parser.add_argument("--knn", type=int, default=16, help="Number of k-nearest neighbors for graph construction")
    parser.add_argument("--kernel", type=str, default="heat", choices=["heat", "inv"], help="Weight kernel type")
    parser.add_argument("--steps-per-epoch", type=int, default=200, help="Training steps per epoch")
    parser.add_argument("--compute-true-eigs", action="store_true", help="Compute ground-truth generalized eigenvalues during training (dense; practical only for smaller problems)")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load geometry and sample the point cloud
    geom, sid = load_mesh_npy(args.base, args.idx)
    faces = geom.faces
    assert faces is not None, "faces are missing: uniform sampling on the mesh surface requires triangle face indices"
    points = sample_points_on_mesh(geom.verts, faces, args.npoints, seed=args.seed)
    points = zero_center_and_scale(points)

    # Build point-cloud operators and form `A`, `M` (convert sparse to dense for training)
    L_sp, M_sp = point_cloud_laplacian(points)
    # def filter_larger(L: sp.csr_matrix, threshold: float = 3):
    #     """Make outliers small"""
    #     abs_data = np.abs(L.data)
    #     mean = abs_data.mean()
    #     sigma = (abs_data - mean).std()
    #     thre = threshold * sigma + mean
    #     L.data[abs_data > thre] = thre * np.sign(L.data[abs_data > thre])
    #     return L

    # L_sp = filter_larger(L_sp, 3)
    print(f"Stat L_sp: max={L_sp.max():.3f} min={L_sp.min():.3f} mean={np.abs(L_sp.data).mean():.3f}, std={np.abs(L_sp.data).std():.3f}")
    print(f"Stat M_sp: max={M_sp.max():.3f} min={M_sp.min():.3f} mean={M_sp.mean():.3f}")

    L_bal, M_bal = balance_stiffness(L_sp, M_sp, args.delta, args.k)
    print(f"Stat L_bal: max={L_bal.max():.3f} min={L_bal.min():.3f} mean={L_bal.mean():.3f}")
    print(f"Stat M_bal: max={M_bal.max():.3f} min={M_bal.min():.3f} mean={M_bal.mean():.3f}")
    # Convert to dense NumPy arrays (avoids false positives from type checkers)
    A_np = np.asarray(ensure_numpy(L_bal), dtype=np.float32)
    M_np = np.asarray(ensure_numpy(M_bal), dtype=np.float32)

    # Build the Lightning model and trainer
    model = SubspaceSolverLit(
        A_np=A_np,
        M_np=M_np,
        points=points,
        k=args.k,
        batch_size=args.batch,
        lr=args.lr,
        max_epochs=args.epochs,
        comp_weight=args.comp_weight,
        kkt_weight=args.kkt_weight,
        grad_clip_norm=1.0,
        plot_every=args.plot_every,
        steps_per_epoch=args.steps_per_epoch,
        compute_true_eigs=args.compute_true_eigs,
    )

    accelerator = "gpu" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1,
        log_every_n_steps=10,
    )

    print(
        f"Loaded id='{sid}' | verts={geom.verts.shape} faces={faces.shape} | N={A_np.shape[0]} | device={accelerator}"
    )

    trainer.fit(model)

    # Optional visualization after training
    if args.scatter and plt is not None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        # Use `scatter3D` to satisfy the type checker's expected signature (`x`, `y`, `z` components)
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap="viridis")
        ax.set_title("Sampled points (uniform on mesh surface)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
