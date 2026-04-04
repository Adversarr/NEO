import math
import torch
from torch import nn
from torch.nn import functional as F

from g2pt.metrics.mse import MSELoss
from g2pt.metrics.rl1 import RelL1Loss
from g2pt.metrics.rmse import RelMSELoss
from g2pt.metrics.rrmse import RootRelMSELoss

def _inner(x, y, mass):
    if mass is None:
        return torch.sum(x * y, dim=1, keepdim=True)  # [B, 1, C]
    else:
        return torch.sum(x * y * mass, dim=1, keepdim=True)

class SpanLoss(nn.Module):
    def __init__(self, use_root: bool = True, norm: int=2, epsilon: float = 1e-6):
        """
        SpanLoss computes the distance between two set of vectors by projecting the predicted onto the target basis.

        If use_root is True, it uses RootRelMSELoss, otherwise it uses RelMSELoss.

        Args:
            use_root (bool): If True, uses RootRelMSELoss, otherwise uses RelMSELoss.
        """
        super().__init__()
        self.epsilon = epsilon
        if norm == 2:
            if use_root:
                self.internal_loss = RootRelMSELoss(channel_wise=True)
            else:
                self.internal_loss = RelMSELoss(channel_wise=True)
        else:
            self.internal_loss = RelL1Loss(channel_wise=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        target_already_normalized: bool = False,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the distance between the predicted and target basis (subspace's basis)

        assuming target is the ground truth, and
        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels.
            target (torch.Tensor): Target tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels.
            mass (torch.Tensor | None): Optional tensor of shape (B, N, 1) representing the mass of each point. If None, no mass is applied.
            target_already_normalized (bool): If True, assumes target is already normalized.

        Returns:
            torch.Tensor: the loss value.
        """
        # normalize the target basis if not already normalized.
        if not target_already_normalized:
            inv_target_norm = torch.rsqrt(_inner(target, target, mass) + self.epsilon)  # [B, 1, Ct]
            target = target * inv_target_norm

        # atb: the coefficients on the target basis of pred vectors.
        if mass is None:
            atb = torch.bmm(pred.transpose(1, 2), target)  # [B, Cp, Ct].
        else:
            atb = torch.bmm(pred.transpose(1, 2), target * mass)

        # reconstruct the predicted points from the target basis
        recon = torch.bmm(target, atb.transpose(1, 2))  # [B, N, Ct] * [B, Ct, Cp] = [B, N, Cp]
        return self.internal_loss(pred=recon, target=pred, mass=mass)

class InverseSpanLoss(nn.Module):
    def __init__(self, use_root: bool = True, norm: int = 2, epsilon: float = 1e-6):
        super().__init__()
        self.spl = SpanLoss(use_root=use_root, norm=norm, epsilon=epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mass: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        return self.spl(pred=target, target=pred, mass=mass, *args, **kwargs)

class BidiSpanLoss(nn.Module):
    def __init__(self, use_root: bool = True, norm: int = 2, epsilon: float = 1e-6):
        """
        BidiSpanLoss computes the bidirectional distance of basis, similar to SpanLoss, but it computes the distance in both directions.

        If use_root is True, it uses RootRelMSELoss, otherwise it uses RelMSELoss.

        Args:
            use_root (bool): If True, uses RootRelMSELoss, otherwise uses RelMSELoss.
        """
        super().__init__()
        self.epsilon = epsilon
        self.spl = SpanLoss(use_root=use_root, norm=norm, epsilon=epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mass: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        """
        Evaluate the distance between the predicted and target basis (subspace's basis)

        assuming target is the ground truth, and
        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels.
            target (torch.Tensor): Target tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels.
            mass (torch.Tensor | None): Optional tensor of shape (B, N, 1) representing the mass of each point. If None, no mass is applied.

        wish <target, mass * target> = 1, and <pred, mass * pred> = 1 (for each channel)

        Returns:
            torch.Tensor: the loss value.
        """

        # normalize the input tensors
        if mass is None:
            pred = pred / (torch.linalg.norm(pred, dim=1, keepdim=True) + self.epsilon)  # Normalize the predicted basis
            target = target / (torch.linalg.norm(target, dim=1, keepdim=True) + self.epsilon)  # Normalize the target basis
        else:
            inv_pred_norm = torch.rsqrt(torch.sum(pred * pred * mass, dim=1, keepdim=True) + self.epsilon)
            inv_targ_norm = torch.rsqrt(torch.sum(target * target * mass, dim=1, keepdim=True) + self.epsilon)
            pred = pred * inv_pred_norm  # Normalize the predicted basis
            target = target * inv_targ_norm

        # Compute the span loss in both directions
        pred_targ = self.spl(pred, target, mass, target_already_normalized=True)
        targ_pred = self.spl(target, pred, mass, target_already_normalized=True)
        return pred_targ + targ_pred


class OrthogonalLoss(nn.Module):
    def __init__(self):
        """
        OrthogonalLoss computes the orthogonality loss of a set of vectors.
        It measures how close the vectors are to being orthogonal.
        """
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        mass: torch.Tensor | None,
        eye: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the orthogonality of the predicted vectors.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels.

        Returns:
            torch.Tensor: Orthogonality loss of the point cloud.
        """
        if mass is None:
            pred_m = pred
        else:
            pred_m = pred * mass
        inner = torch.bmm(pred.transpose(1, 2), pred_m)  # inner product [B, C, C]
        eye = torch.eye(pred.size(-1), device=pred.device).expand_as(inner)
        return F.mse_loss(inner, eye)

class LstsqLoss(nn.Module):
    def __init__(self, use_root: bool = True, norm: int = 2, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        if norm == 2:
            if use_root:
                self.internal_loss = RootRelMSELoss(channel_wise=True)
            else:
                self.internal_loss = RelMSELoss(channel_wise=True)
        else:
            self.internal_loss = RelL1Loss(channel_wise=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mass: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        A = pred # [B, N, C1]
        B = target # [B, N, C2]
        if mass is not None:
            sqrt_mass = torch.sqrt(mass + self.epsilon)
            A_weighted = A * sqrt_mass  # [B, N, C1]
            B_weighted = B * sqrt_mass  # [B, N, C2]
            X = torch.linalg.lstsq(A_weighted, B_weighted).solution  # [B, C1, C2]
        else:
            X = torch.linalg.lstsq(A, B).solution
        B_recon = torch.bmm(A, X)  # [B, N, C2]
        return self.internal_loss(pred=B_recon, target=B, mass=mass)



class ProjectionLoss_Old(nn.Module):
    def __init__(self, use_root: bool = True, norm: int = 2, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.use_root = use_root

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        # unscale the pred and target
        if mass is not None:
            mass = mass.float()
            sqrt_M = torch.sqrt(mass + self.epsilon)  # [B, N, 1]
            pred = pred * sqrt_M  # [B, N, C1]
            target = target * sqrt_M  # [B, N, C2]
            # normalize the target basis by its norm, since the mass matrix may be normalzied, such that the 
            # target basis is not well normalized.
            pred = pred / (pred.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C1]
            target = target / (target.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C2]

        # normal way
        upper = min(pred.shape[-1], target.shape[-1]) * 2
        utv = torch.bmm(pred.mT, target)  # [B, C1, C2]
        err = upper - 2 * torch.sum(torch.square(utv), dim=(-1, -2))  # [B]
        if self.use_root:
            loss = torch.sqrt(err + self.epsilon).mean()  # Return the mean error across the batch
        else:
            loss = err.mean()  # Return the mean error across the batch
        return loss

class ProjectionLoss(nn.Module):
    def __init__(self, use_root: bool = True, reduction: str = "mean", epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.use_root = use_root 
        self.reduction = reduction

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Calculates the subspace distance using efficient dot products.
        Goal: Ensure subspace spanned by 'target' is contained within subspace spanned by 'pred'.
        
        Args:
            pred: [B, N, m] - Neural network output (prediction)
            target: [B, N, n] - Target basis vectors
            mass: [B, N, 1] - Optional mass matrix for generalized inner product
        """
        pred = pred.float()
        target = target.float()

        # 1. Generalized Unscaling (Inner Product Space adjustment)
        if mass is not None:
            mass = mass.float()
            sqrt_M = torch.sqrt(mass + self.epsilon)
            pred = pred * sqrt_M 
            target = target * sqrt_M 

        # 2. Orthonormalization / Normalization
        # We normalize columns to unit length to treat them as direction vectors.
        # Note: We assume 'pred' vectors are roughly orthogonal to each other if m > n, 
        # but strictly we only need unit length here to compute cosine similarity.
        pred_norm = pred.norm(dim=1, keepdim=True) + self.epsilon
        target_norm = target.norm(dim=1, keepdim=True) + self.epsilon
        
        pred = pred / pred_norm
        target = target / target_norm

        # 3. Efficient Projection Calculation
        # Shape: [B, m, n]
        # Calculates cos(theta) between every pred vector and every target vector
        cosine_similarity_matrix = torch.bmm(pred.mT, target)

        # 4. Compute Projection Energy per Target Vector
        # We want to know how much of each 'target' vector is captured by the 'pred' subspace.
        # Since pred vectors are (assumed) orthogonal, energy is sum of squared projections.
        # Shape: [B, n]
        projection_energies = torch.sum(torch.square(cosine_similarity_matrix), dim=1)

        # 5. Compute Residual (Unexplained Energy)
        # Ideally, energy is 1.0 if fully contained. 
        # We clamp to avoid numerical issues (sqrt of negative) when energy > 1 (due to float error).
        residuals = 1.0 - projection_energies
        residuals = torch.relu(residuals)

        # 6. Loss Calculation
        # sqrt(residual) converts "Squared Error" (Energy) to "Euclidean Distance" (L2 Norm)
        # This matches NeurKItt's behavior: sum/mean of distances, not squared distances.
        if self.use_root:
            vector_losses = torch.sqrt(residuals + self.epsilon)
        else:
            vector_losses = residuals
        vector_losses = torch.sum(vector_losses, dim=1) # [B]

        # 7. Reduction
        if self.reduction == "mean":
            return vector_losses.mean()
        elif self.reduction == "sum":
            return vector_losses.sum()
        else:
            return vector_losses.mean() # Default


class ProjectionEstimationLoss(nn.Module):
    def __init__(self, estim_dim: int=256, use_root: bool = True, norm: int = 2, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.estim_dim = estim_dim
        if norm == 2:
            if use_root:
                self.internal_loss = RootRelMSELoss(channel_wise=True)
            else:
                self.internal_loss = RelMSELoss(channel_wise=True)
        else:
            self.internal_loss = RelL1Loss(channel_wise=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        # mask removed
        *args, **kwargs
    ) -> torch.Tensor:
        trial = torch.randn(pred.size(0), pred.size(1), self.estim_dim, device=pred.device, dtype=pred.dtype)

        # coordinates on each basis.
        if mass is not None:
            trial_pred = torch.bmm((pred * mass).mT, trial)
            trial_target = torch.bmm((target * mass).mT, trial)
        else:
            trial_pred = torch.bmm(pred.mT, trial)
            trial_target = torch.bmm(target.mT, trial)

        trial_pred_recon = torch.bmm(pred, trial_pred)
        trial_target_recon = torch.bmm(target, trial_target)

        return self.internal_loss(
            pred=trial_pred_recon,
            target=trial_target_recon,
            mass=mass,
            # mask removed
        )

class GrassmannDistance(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        if pred.shape[-1] > target.shape[-1]:
            U, V = target, pred
        else:
            U, V = pred, target

        if mass is not None:
            A_M = torch.bmm(U.mT, V * mass)  # [B, C1, C2]
        else:
            A_M = torch.bmm(U.mT, V)

        sigma = torch.linalg.svdvals(A_M)  # [B, min(C1, C2)]
        sigma = torch.clip(sigma, -1, 1)
        theta = torch.acos(sigma)  # [B, min(C1, C2)]
        distance = torch.linalg.norm(theta, dim=1)  # [B]
        return distance.mean()  # Return the mean distance across the batch

class SelfDistance(nn.Module):
    def __init__(self, epsilon: float = 1e-6, asymmetric: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.asymmetric = asymmetric

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        y_original: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encourage the pred to be non-degenerate.

        pred: (b, n, c), batch size, number of points, number of funcs.
        mass: (b, n, 1), mass matrix.
        """
        y_original = y_original.float()
        mass = mass.float()
        if mass is not None:
            pred_m = y_original * mass
        else:
            pred_m = y_original

        # whether to use our asymmetric loss.
        pred_trial = y_original.mT.detach() if self.asymmetric else y_original.mT     # (b, c, n)
        vol = torch.clamp(torch.sum(mass, dim=1, keepdim=True), min=self.epsilon)     # (b, 1, 1)
        ortho = torch.tril(torch.bmm(pred_trial, pred_m), diagonal=1) / vol
        return ortho.abs().mean()

class BiorthoLoss(nn.Module):
    """Encourage the p_basis and q_basis to be orthogonal."""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        p_basis,
        q_basis,
        mass: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encourage the p_basis and q_basis to be orthogonal.

        p_basis: (b, n, c), batch size, number of points, number of funcs.
        q_basis: (b, n, c), batch size, number of points, number of funcs.
        mass: (b, n, 1), mass matrix.
        """

        if mass is not None:
            p_basis = p_basis / torch.sqrt(torch.sum(p_basis.square() * mass, dim=1, keepdim=True))
            q_basis = q_basis / torch.sqrt(torch.sum(q_basis.square() * mass, dim=1, keepdim=True))
            cov = torch.einsum("bni,bnj,bn->bij", p_basis, q_basis, mass.squeeze(-1))
        else:
            p_basis = p_basis / torch.sqrt(torch.sum(p_basis.square(), dim=1, keepdim=True))
            q_basis = q_basis / torch.sqrt(torch.sum(q_basis.square(), dim=1, keepdim=True))
            cov = torch.einsum("bni,bnj->bij", p_basis, q_basis)

        return torch.tril(cov, diagonal=1).square().mean()


class ProjectionLoss_NeurKItt(nn.Module):
    """https://github.com/smart-JLuo/NeurKItt/blob/main/loss.py"""

    def __init__(self, use_root=True, num_type="real", reduction="mean", p=2, dim=1, epsilon=1e-6):
        super(ProjectionLoss_NeurKItt, self).__init__()
        self.use_root = use_root
        self.num_type = num_type
        self.reduction = reduction
        self.p = p
        self.dim = dim
        self.epsilon = epsilon

        if self.num_type == "complex":
            self.trans = torch.adjoint
        else:
            self.trans = torch.transpose

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        """
        Q: [batch_size,dim,k]
        V: [batch_size,dim,K] # K > k
        caculate sum(V-Q@Q*V)
        """
        # unscale the pred and target
        if mass is not None:
            mass = mass.float()
            sqrt_M = torch.sqrt(mass + self.epsilon)  # [B, N, 1]
            pred = pred * sqrt_M  # [B, N, C1]
            target = target * sqrt_M  # [B, N, C2]
            # normalize the target basis by its norm, since the mass matrix may be normalzied, such that the
            # target basis is not well normalized.
            pred = pred / (pred.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C1]
            target = target / (target.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C2]

        Q = pred
        V = target

        assert Q.shape[-2] == V.shape[-2], (
            f"Shape error! Q shape [{Q.shape[-2]}] must match V shape [{V.shape[-2]}] in dimension -2"
        )

        Qt = Q.mT
        QtV = torch.bmm(Qt, V)
        QQtV = torch.bmm(Q, QtV)

        result = V - QQtV
        if self.use_root:
            norm = torch.norm(result, p=self.p, dim=self.dim)
        else:
            norm = torch.sum(result ** self.p, dim=self.dim)

        loss = torch.sum(norm, dim=-1)

        if self.reduction == "mean":
            result = torch.mean(loss)
        elif self.reduction == "sum":
            result = torch.sum(loss)
        return result


class PrincipalAngle(nn.Module):
    """
    Caculate the angle between two subspace.
    type[Option]: biggest or smallest angle between two subspace

    WARINING: IF YOU NEED COMPUTE GRADIENT, PLEASE TURN OFF THE CLIP
    """

    def __init__(self, angle_type="biggest", reduction="mean", clip_value=True):
        super(PrincipalAngle, self).__init__()
        self.angle_type = angle_type
        self.reduction = reduction
        self.clip_value = clip_value
        self.compare = torch.max if self.angle_type == "biggest" else torch.min
        self.epsilon = 1e-6

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        """
        Q,V: base vectors which make up the subspace
        """
        # unscale the pred and target
        if mass is not None:
            mass = mass.float()
            sqrt_M = torch.sqrt(mass + self.epsilon)  # [B, N, 1]
            pred = pred * sqrt_M  # [B, N, C1]
            target = target * sqrt_M  # [B, N, C2]
            # normalize the target basis by its norm, since the mass matrix may be normalzied, such that the
            # target basis is not well normalized.
            pred = pred / (pred.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C1]
            target = target / (target.norm(dim=1, keepdim=True) + self.epsilon)  # [B, N, C2]

        with torch.autocast(device_type = pred.device.type, enabled=False):
            Q = pred.float() if pred.dtype != torch.float32 else pred
            V = target.float() if pred.dtype != torch.float32 else pred

            _, values, _ = torch.linalg.svd(torch.bmm(torch.transpose(Q, -2, -1), V))

            if self.clip_value:
                values = torch.clamp(values, min=-1, max=1)

            angles = torch.acos(values)

            angle, _ = self.compare(angles, dim=-1)

        if self.reduction == "mean":
            result = torch.mean(angle)
        elif self.reduction == "sum":
            result = torch.sum(angle)
        else:
            result = angle

        return result
