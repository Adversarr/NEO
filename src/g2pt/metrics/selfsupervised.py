import torch
from torch import nn
from torch.nn import functional as F
from g2pt.utils.sparse import GraphSpmv, NativeSpmv, SymmSparseCSRMatmul
from g2pt.utils.common import ensure_numpy
import numpy as np

class SelfSupervisedLoss(nn.Module):
    def __init__(self, lamb_comp=0.5, backend="native"):
        super().__init__()
        self.lamb_comp = lamb_comp
        if backend == "stable":
            self.spmv = GraphSpmv()
        elif backend == "native":
            self.spmv = NativeSpmv()
        else:
            raise ValueError(f"Unknown SPMM backend: {backend}")

    def forward(
        self,
        x_hat: torch.Tensor,  # A^-1 rhs
        rhs: torch.Tensor,  # rhs
        sysmat: tuple[torch.Tensor, torch.Tensor],  # A
        subspace_vectors: torch.Tensor,  # [B, N, q_dim]
        mass: torch.Tensor,
        sysmat_csr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Self-supervised loss matching demo: compliance + KKT energy.

        It is responsible for the GEVP:
            A x = lamb M x

        You should bias A to be positive-definite enough such that the GEVP is well conditioned.

        Shapes and semantics:
        - sol: [B, N, nsol] predicted responses (x̂). nsol = number of RHS samples per shape (demo's batch size).
        - rhs: [B, N, nsol] random right-hand sides per shape (b in demo).
        - sparse_stiffness: [B*N, B*N] CSR of stiffness.
        - sparse_mass: [B*N, B*N] CSR mass matrix (optional).
        - lumped_mass: [B, N, 1] per-shape lumped mass weights; used to compute vol = ∑ w_i for each shape.

        Returns:
        - torch.Tensor: scalar loss.
        """

        # Infer shapes from inputs
        B, N, nsol = x_hat.shape  # batch size, number of points, number of RHS per shape
        x_hat = x_hat.contiguous()
        rhs = rhs.contiguous()  # avoid overwriting B; keep rhs explicit

        # Ensure mass has shape [B, N, 1] to match broadcasting semantics in loss
        if mass.dim() == 2:
            mass = mass.unsqueeze(-1)

        # Mass-weighted loads per shape and RHS
        Mb = mass * rhs  # [B, N, nsol]
        vol = mass.sum(dim=1)  # [B, 1]


        # Sparse stiffness times predicted response, done in flattened space then reshaped back
        with torch.autocast(device_type=x_hat.device.type, enabled=False):
            x_flat = x_hat.view(B * N, nsol)
            x_flat = x_flat.float() if x_flat.dtype != torch.float32 else x_flat
            if sysmat_csr is None:
                A_ind, A_val = sysmat  # (2, nnz) (nnz, )
                A_val_in = A_val.float() if A_val.dtype != torch.float32 else A_val
                Ax_hat: torch.Tensor = self.spmv(x_flat, A_ind, A_val_in).view(B, N, nsol)  # type:ignore
            else:
                assert sysmat_csr.is_sparse_csr, "sysmat_csr must be a CSR sparse matrix"
                sysmat_csr = sysmat_csr.float() if sysmat_csr.dtype != torch.float32 else sysmat_csr
                Ax_hat: torch.Tensor = SymmSparseCSRMatmul.apply(sysmat_csr, x_flat).view(B, N, nsol)  # type:ignore

        # KKT Residual Part
        a_energy = (Ax_hat * x_hat).sum(dim=1)  # [B, nsol]
        load = (Mb * x_hat).sum(dim=1)  # [B, nsol]

        sigma = (load / a_energy.clamp(min=1e-4)) # [B, nsol]

        kkt_energy = (0.5 * a_energy * sigma - load) * sigma / vol  # [B, nsol]
        kkt = kkt_energy.mean()

        # # # Residual Part, optional
        # q_dim = subspace_vectors.shape[-1]
        # r = Ax_hat - Mb # (b, n, nsol)
        # # real projection should have r ortho to q space
        # space_vecs_norm = torch.sqrt(torch.sum(subspace_vectors.square() * mass, dim=1, keepdim=True))  # (b, 1, q_dim)
        # qr = torch.bmm(r.mT, (subspace_vectors / space_vecs_norm)) / vol.unsqueeze(-1) # (b, nsol, q_dim)
        # kkt_res_loss = F.mse_loss(qr, torch.zeros_like(qr))
        # # kkt = kkt_res_loss
        # kkt = kkt # + kkt_res_loss

        # Loss weights
        w_kkt = 1 - self.lamb_comp
        w_compliance = self.lamb_comp

        # Compliance Part (align with demo): comp = - mean( (Mb * x̂).sum / vol )
        comp_batch = sigma * (Mb * x_hat.contiguous()).sum(dim=1) / vol  # [B, nsol]
        comp = -comp_batch.mean()

        # Total Loss
        loss = w_compliance * comp + w_kkt * kkt
        return loss