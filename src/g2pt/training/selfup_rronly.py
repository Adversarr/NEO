from torch.nn import ModuleList
from g2pt.metrics import get_metric
from g2pt.metrics.selfsupervised import SelfSupervisedLoss
from g2pt.utils.ortho_operations import qr_orthogonalization
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
import numpy as np
import torch
from torch import nn
from torch import distributed as dist, optim
import math
from copy import deepcopy
from g2pt.metrics.span import ProjectionLoss, SelfDistance
from g2pt.neuralop.model import get_model, get_sol_model
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_with_gt
from g2pt.utils.sparse import GraphSpmv, NativeSpmv, SymmSparseCSRMatmul
from g2pt.training.common import create_optimizer_and_scheduler, load_partial_state_dict_strict, load_params_state_dict_strict


class RQLoss(nn.Module):
    def __init__(self, use_sqrt: bool = True) -> None:
        super().__init__()
        self.use_sqrt = use_sqrt
        if use_sqrt:
            print("Warning: use_sqrt is True, which may cause mathematical error!")

    def forward(self,
        e_i: torch.Tensor,  # [B, N, q_dim]
        mass: torch.Tensor,
        sysmat_csr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the Rayleigh quotient loss.
        """
        B, N, q_dim = e_i.shape
        # Sparse stiffness times predicted response, done in flattened space then reshaped back
        with torch.autocast(device_type=e_i.device.type, enabled=False):
            e_flat = e_i.reshape(B * N, q_dim)
            e_flat = e_flat.float() if e_flat.dtype != torch.float32 else e_flat
            A_e_i: torch.Tensor = SymmSparseCSRMatmul.apply(sysmat_csr, e_flat).view(B, N, q_dim)  # type:ignore

        rq_diag = (e_i * A_e_i).sum(dim=1)  # [B, q_dim]
        if self.use_sqrt:
            rq_diag = rq_diag.clamp_min(1e-12).sqrt()
        return rq_diag.mean()


##### Model #####

class SelfSupervisedRQTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.targ_dim: int = cfg.get("targ_dim", 16)
        self.targ_dim_model: int = cfg.get("targ_dim_model", self.targ_dim)
        self.balancing_delta: float = cfg.get("balancing_delta", 1)
        self.model = get_model(3, 3, self.targ_dim_model, cfg.model)
        self.model.reset_parameters()  # Reset parameters of the model.

        self.freeze_pretrained = cfg.freeze_pretrained

        pretrained = getattr(cfg, 'ckpt_pretrain', None) or cfg.ckpt_pretrain
        has_pretrain = False
        if pretrained:
            print(f"🎉 Load pretrained model params (strict partial) from {pretrained}")
            try:
                load_partial_state_dict_strict(self.model, ckpt_path=pretrained, key_prefix='model.')
                has_pretrain = True
            except Exception as e:
                print(f"⚠️ Failed to load pretrained model params: {e}")

        # Training
        epochs: int = cfg.get("epochs", 500)
        self.epochs = epochs

        # Read optimizer/scheduler from Hydra cfg (exp/conf/optim.yaml)
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler

        self.loss = RQLoss(use_sqrt=cfg.rq_use_sqrt)
        self.rq_weight = cfg.rq_weight
        self.n_samples = cfg.get("n_samples", 256)
        self.proj_loss = ProjectionLoss()

        if not has_pretrain:
            print("⚠️ No pretrained model params load!")

        if self.freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("🧊 Freezing pretrained model.")

        # Pretraining Stage
        self.losses = ModuleList()
        self.weights = []
        for metric_args in cfg.training_metrics:
            kwargs = metric_args.get("kwargs", {})
            self.losses.append(get_metric(name=metric_args["name"], **kwargs))
            self.weights.append(metric_args.get("weight", 1.0))

        self.enable_ema = cfg.get("enable_ema", False)
        self.ema_decay = cfg.get("ema_decay", cfg.get("ema_momentum", 0.999))
        self.ema_warmup_steps = cfg.get("ema_warmup_steps", 0)
        self.ema_update_every = cfg.get("ema_update_every", 1)
        self._ema_last_step = -1
        if self.enable_ema:
            self.ema_model = deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self.ema_model.eval()
        self.warmup_steps_rq_loss = cfg.get("warmup_steps_rq_loss", 200)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.enable_ema:
            if self.global_step != self._ema_last_step and (self.global_step % self.ema_update_every == 0):
                decay = self._compute_ema_decay(self.global_step)
                self._ema_update_model(decay)
                self._ema_last_step = self.global_step

    def _compute_ema_decay(self, step: int) -> float:
        if self.ema_warmup_steps and self.ema_warmup_steps > 0:
            return 1.0 - (1.0 - self.ema_decay) * (1.0 - math.exp(-float(step) / float(self.ema_warmup_steps)))
        return self.ema_decay

    def _ema_update_model(self, decay: float) -> None:
        with torch.no_grad():
            if not dist.is_initialized() or dist.get_rank() == 0:
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
                for ema_buf, model_buf in zip(self.ema_model.buffers(), self.model.buffers()):
                    ema_buf.data.copy_(model_buf.data)
            if dist.is_initialized():
                for ema_param in self.ema_model.parameters():
                    dist.broadcast(ema_param.data, src=0)
                for ema_buf in self.ema_model.buffers():
                    dist.broadcast(ema_buf.data, src=0)
        self.ema_model.eval()

    def forward(self, x: torch.Tensor, mass: torch.Tensor, return_original: bool=False):
        f_i = self.model(x, fx=x, mass=mass)
        e_i = qr_orthogonalization(f_i, mass) # (b, n, q_dim)
        return (e_i, f_i) if return_original else e_i

    def get_current_rq_weight(self):
        if self.global_step < self.warmup_steps_rq_loss and self.warmup_steps_rq_loss > 0:
            return self.rq_weight * self.global_step / self.warmup_steps_rq_loss
        return self.rq_weight

    def training_step(self, batch, batch_idx):
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        q_basis = self(x, mass) # (b,n,q_dim)
        bs = q_basis.shape[0]
        # For pretrain part
        pretrain_x = batch['pretrain_points']
        pretrain_y = batch['pretrain_evecs']
        pretrian_mass = batch['pretrain_mass']
        pretrain_y_original = self.model(pretrain_x, fx=pretrain_x, mass=pretrian_mass)
        pretrain_y_hat = qr_orthogonalization(pretrain_y_original, pretrian_mass)

        with torch.autocast(x.device.type, enabled=False):
            # 1. Self-supervised
            sysmat_csr = batch.get("stiff_csr", None)
            mass = mass.float()
            self_supervised = self.loss(
                e_i=q_basis,
                mass=mass,
                sysmat_csr=sysmat_csr,
            )
            total_loss = self_supervised * self.get_current_rq_weight()
            with torch.no_grad():
                try:
                    B, N, q_dim = q_basis.shape
                    e_flat = q_basis.reshape(B * N, q_dim).float()
                    A_e_i = SymmSparseCSRMatmul.apply(sysmat_csr, e_flat).view(B, N, q_dim)  # type:ignore
                    rq_diag = (q_basis * A_e_i).sum(dim=1).clamp_min(1e-12)
                    self.log("Train/rq_diag_mean", rq_diag.mean(), batch_size=bs)
                    self.log("Train/rq_diag_min", rq_diag.min(), batch_size=bs)
                    self.log("Train/rq_diag_max", rq_diag.max(), batch_size=bs)
                except Exception:
                    pass

            # 2. Pretrain mix
            for loss_fn, weight in zip(self.losses, self.weights):
                # TODO: Metrics may accept `mask`; removed here since masking is deprecated.
                loss_value = loss_fn(
                    pred=pretrain_y_hat,
                    target=pretrain_y,
                    mass=pretrian_mass,
                    y_original=pretrain_y_original,
                )
                if weight > 0:
                    total_loss = total_loss + loss_value * weight
                self.log(f"Train/PRE-{type(loss_fn).__name__}", loss_value, batch_size=bs)

            self.log("Train/mesh_size", x.shape[1], batch_size=bs)
            self.log("Train/loss", total_loss, prog_bar=True, batch_size=bs)
            self.log("Train/self_supervise", self_supervised, prog_bar=True, batch_size=bs)

            if batch_idx == 0 and self.global_rank == 0:
                log_dict = {}
                try:
                    ranks = torch.linalg.matrix_rank(q_basis)
                    log_dict["Train/rank"] = ranks.float().mean().item()
                except Exception as e:
                    self.print(f"Warning(train): Rank validation failed: {e}")
                try :
                    q = q_basis
                    subspace0 = q[0]
                    x0 = x[0].detach().cpu()  # (n, 3)
                    m0 = mass[0].detach().cpu().numpy()
                    evec, evec_gt = solve_gev_from_subspace_with_gt(
                        subspace0, x0, mass=mass[0], k=self.targ_dim, delta=self.balancing_delta
                    )

                    scores = np.abs(outer_cosine_similarity(evec, evec_gt, M=m0)) # (c, d)
                    max_scores = np.max(np.abs(scores), axis=0)  # [d]

                    log_dict["Train/evec_max_scores_mean"] = 1 - max_scores.mean()
                    log_dict["Train/evec_max_scores_max"] = 1 - max_scores.max()
                    log_dict["Train/evec_max_scores_min"] = 1 - max_scores.min()
                    log_dict["Train/evec_max_scores_median"] = 1 - np.median(max_scores)
                    log_dict["Train/evec_max_scores_mismatch"] = 1 - np.trace(np.abs(scores)) / scores.shape[1]
                    with torch.no_grad():
                        evec_gt = torch.from_numpy(evec_gt).to(dtype=mass.dtype, device=mass.device)
                        proj_loss = self.proj_loss(q[[0]], evec_gt.unsqueeze(0), mass[[0]])
                    log_dict["Train/proj_loss"] = proj_loss.item()
                except Exception as e:
                    self.print(f"Warning(train): Eigenvalue validation failed: {e}")
                self.log_dict(log_dict, rank_zero_only=True, batch_size=bs)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        q_basis = self(x, mass) # (b,n,q_dim)
        bs = q_basis.shape[0]

        # For pretrain part
        pretrain_x = batch['pretrain_points']
        pretrain_y = batch['pretrain_evecs']
        pretrian_mass = batch['pretrain_mass']
        pretrain_y_original = self.model(pretrain_x, fx=pretrain_x, mass=pretrian_mass)
        pretrain_y_hat = qr_orthogonalization(pretrain_y_original, pretrian_mass)

        with torch.autocast(x.device.type, enabled=False):
            sysmat_csr = batch.get("stiff_csr", None)

            mass = mass.float()
            q_basis = q_basis.float()
            self_supervised = self.loss(
                e_i=q_basis,
                mass=mass,
                sysmat_csr=sysmat_csr,
            )
            total_loss = self_supervised * self.get_current_rq_weight()
            with torch.no_grad():
                try:
                    B, N, q_dim = q_basis.shape
                    e_flat = q_basis.reshape(B * N, q_dim).float()
                    A_e_i = SymmSparseCSRMatmul.apply(sysmat_csr, e_flat).view(B, N, q_dim)  # type:ignore
                    rq_diag = (q_basis * A_e_i).sum(dim=1).clamp_min(1e-12)
                    self.log("Val/rq_diag_mean", rq_diag.mean(), batch_size=bs)
                    self.log("Val/rq_diag_min", rq_diag.min(), batch_size=bs)
                    self.log("Val/rq_diag_max", rq_diag.max(), batch_size=bs)
                except Exception:
                    pass

            # 2. Pretrain mix
            for loss_fn, weight in zip(self.losses, self.weights):
                # TODO: Metrics may accept `mask`; removed here since masking is deprecated.
                loss_value = loss_fn(
                    pred=pretrain_y_hat,
                    target=pretrain_y,
                    mass=pretrian_mass,
                    y_original=pretrain_y_original,
                )
                if weight > 0:
                    total_loss = total_loss + loss_value * weight
                self.log(f"Val/PRE-{type(loss_fn).__name__}", loss_value, batch_size=bs)

        self.log("Val/total", total_loss, batch_size=bs)
        self.log("Val/self_supervise", self_supervised, batch_size=bs)

        if batch_idx == 0 and self.global_rank == 0:
            log_dict = {}
            try:
                ranks = torch.linalg.matrix_rank(q_basis)
                log_dict["Val/rank"] = ranks.float().mean().item()
            except Exception as e:
                self.print(f"Warning(val): Rank validation failed: {e}")
            try:
                q = q_basis

                subspace0 = q[0]
                x0 = x[0].detach().cpu()  # (n, 3)
                m0 = mass[0].detach().cpu().numpy()
                evec, evec_gt = solve_gev_from_subspace_with_gt(
                    subspace0, x0, mass=mass[0], k=self.targ_dim, delta=self.balancing_delta
                )

                scores = np.abs(outer_cosine_similarity(evec, evec_gt, M=m0)) # (c, d)
                max_scores = np.max(np.abs(scores), axis=0)  # [d]

                log_dict["Val/evec_max_scores_mean"] = 1 - max_scores.mean()
                log_dict["Val/evec_max_scores_max"] = 1 - max_scores.max()
                log_dict["Val/evec_max_scores_min"] = 1 - max_scores.min()
                log_dict["Val/evec_max_scores_median"] = 1 - np.median(max_scores)
                log_dict["Val/evec_max_scores_mismatch"] = 1 - np.trace(np.abs(scores)) / scores.shape[1]
                with torch.no_grad():
                    evec_gt = torch.from_numpy(evec_gt).to(dtype=mass.dtype, device=mass.device)
                    proj_loss = self.proj_loss(q[[0]], evec_gt.unsqueeze(0), mass[[0]])
                log_dict["Val/proj_loss"] = proj_loss.item()
                for i in range(self.targ_dim):
                    log_dict[f"ValDetails/evec_max_scores_{i}"] = 1 - max_scores[i]
            except Exception as e:
                self.print(f"Warning(val): Eigenvalue validation failed: {e}")
            self.log_dict(log_dict, rank_zero_only=True, batch_size=bs)

        return total_loss

    def configure_optimizers(self):  # type: ignore
        if self.freeze_pretrained:
            raise ValueError("`freeze_pretrained=true` leaves no trainable parameters in SelfSupervisedRQTraining.")
        # Use shared factory to create optimizer and scheduler from cfg
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.model,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
