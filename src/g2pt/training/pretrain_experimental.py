import math
from typing import Tuple

from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
import numpy as np
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn import ModuleList, functional as F

from g2pt.metrics import get_metric
from g2pt.metrics.koleo import KoLeoLoss
from g2pt.metrics.span import InverseSpanLoss, OrthogonalLoss
from g2pt.neuralop.layers.mlps import FeedForwardWithGating, FeedForwardWithGatingConfig
from g2pt.neuralop.model import Transolver2Model, TransolverNeXtModel, get_model
from g2pt.training.common import create_optimizer_and_scheduler
from g2pt.utils.common import roundup, roundup_16
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_with_gt
from g2pt.utils.ortho_operations import qr_orthogonalization
from g2pt.utils.rot import random_rotate_3d


def random_rotate_3d_batched(rot, batch_size: int):
    all_rotates = [random_rotate_3d(rot) for _ in range(batch_size)]
    return np.stack(all_rotates, axis=0) # (batch_size, 3, 3)


def js_div_stable(logits_p, logits_q, reduction="batchmean", eps=1e-8):
    """
    Numerically stable JS divergence: avoids redundant exp/log operations.
    """
    # Compute directly in log space for better precision
    log_p = F.log_softmax(logits_p, dim=-1)  # log(p)
    log_q = F.log_softmax(logits_q, dim=-1)  # log(q)

    # Compute m = (p+q)/2
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    log_m = torch.log(0.5 * (p + q) + eps)

    # Use log_target=True to pass log(p) directly, avoiding redundant log inside kl_div
    kl_pm = F.kl_div(log_m, log_p, reduction=reduction, log_target=True)
    kl_qm = F.kl_div(log_m, log_q, reduction=reduction, log_target=True)

    return 0.5 * (kl_pm + kl_qm)

def kl_div(logits_p, logits_q, reduction="batchmean"):
    # Compute directly in log space for better precision
    log_p = F.log_softmax(logits_p, dim=-1)  # log(p)
    log_q = F.log_softmax(logits_q, dim=-1)  # log(q)

    kl_pm = F.kl_div(log_q, log_p, reduction=reduction, log_target=True)

    return kl_pm

class PretrainTraining_Experimental(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.targ_dim: int = cfg.get("targ_dim", 16)
        self.targ_dim_model: int = cfg.get("targ_dim_model", self.targ_dim)
        self.weight_decay: float = cfg.get("weight_decay", 0.01)
        self.ortho_algorithm = cfg.get("ortho_algorithm", "qr")  # Use QR decomposition for orthogonalization.
        self.muon = cfg.get("muon", False)  # Use Muon Optimizer
        self.model: Transolver2Model | TransolverNeXtModel = get_model(3, 3, self.targ_dim_model, cfg.model) # type: ignore
        self.model.reset_parameters()  # Reset parameters of the model.
        assert cfg.model.name == 'transolver2' or cfg.model.name == 'transolver_next'

        self.epochs = cfg.epochs
        self.masking_algorithm = cfg.get("masking_algorithm", "disable")
        self.masking_dropout_p_lower = cfg.get("masking_dropout_p_lower", 0.0)  # Dropout probability for masking.
        self.masking_dropout_p_upper = cfg.get("masking_dropout_p_upper", 0.5)  # Dropout probability for masking.

        # Read optimizer/scheduler from Hydra cfg (exp/conf/optim.yaml)
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler

        # error metrics:
        self.ortho_loss = OrthogonalLoss()
        self.ispan = InverseSpanLoss(use_root=True, norm=2, epsilon=1e-6)
        self.val_losses = ModuleList()
        self.val_weights = []
        for val_loss_args in cfg.validation_metrics:
            kwargs = val_loss_args.get("kwargs", {})
            self.val_losses.append(get_metric(name=val_loss_args["name"], **kwargs))
            self.val_weights.append(val_loss_args.get("weight", 1.0))

        self.losses = ModuleList()
        self.weights = []
        for metric_args in cfg.training_metrics:
            kwargs = metric_args.get("kwargs", {})
            self.losses.append(get_metric(name=metric_args["name"], **kwargs))
            self.weights.append(metric_args.get("weight", 1.0))

        # regularities:
        self.noise_initial = cfg.get("noise_initial", 0.02)
        self.noise_final = cfg.get("noise_final", 0.0)
        self.noise_ratio = cfg.get("noise_ratio", 0.3)
        self.noise_schedule = cfg.get("noise_schedule", "linear")  # linear or cosine schedule.
        self.random_rotate = cfg.datamod.get("random_rotate", 6.28)
        # Specials
        self.div_weight = cfg.get("kl_weight", 0.2)
        self.koleo_weight = cfg.get("koleo_weight", 0.2)
        self.ewa_alpha_start = cfg.get("ewa_alpha_start", 0.996)
        self.ewa_alpha_end = cfg.get("ewa_alpha_end", 0.9998)
        self.momentum_C: float = cfg.get("momentum_C", 0.999)
        self.student_temperature = cfg.get("student_temperature", 0.1)
        self.teacher_temperature = cfg.get("teacher_temperature", 0.04)
        self.teacher = get_model(3, 3, self.targ_dim_model, cfg.model) # type: ignore
        self.teacher.load_state_dict(self.model.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.koleo = KoLeoLoss()
        self.teacher.eval()
        self.head_dino = nn.Sequential(
            nn.LayerNorm(self.model.out_dim),
            FeedForwardWithGating(self.model.out_dim, self.model.out_dim, FeedForwardWithGatingConfig()),
        )
        self.register_buffer("teacher_C", torch.zeros(self.model.out_dim, dtype=torch.float32))

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EWA update teacher at batch end and sync across ranks"""
        current_epochs = self.current_epoch
        ewa_alpha = self.ewa_alpha_start + (self.ewa_alpha_end - self.ewa_alpha_start) * current_epochs / self.epochs
        self._ema_update_teacher(ewa_alpha)

    def _ema_update_teacher(self, ewa_alpha: float) -> None:
        with torch.no_grad():
            if not dist.is_initialized() or dist.get_rank() == 0:
                for param, teacher_param in zip(self.model.parameters(), self.teacher.parameters()):
                    teacher_param.mul_(ewa_alpha).add_(param, alpha=1 - ewa_alpha)
                for teacher_buf, model_buf in zip(self.teacher.buffers(), self.model.buffers()):
                    teacher_buf.data.copy_(model_buf.data)
            if dist.is_initialized():
                for teacher_param in self.teacher.parameters():
                    dist.broadcast(teacher_param.data, src=0)
                for teacher_buf in self.teacher.buffers():
                    dist.broadcast(teacher_buf.data, src=0)
        self.teacher.eval()

    def _add_noise(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [b, n, 3].
        """
        if not self.training:  # During validation, we do not apply any masking.
            return x, y, mass

        # 1. Add noise to the input point cloud (x)
        current_ratio = max(min(self.current_epoch / (self.epochs * self.noise_ratio), 1.0), 0.0)
        if self.noise_schedule == "linear":
            scale: float = 1 - current_ratio
        elif self.noise_schedule == "cosine":
            scale = 0.5 * (1 + math.cos(math.pi * current_ratio))
        else:
            assert self.noise_schedule == "disable"
            scale = 0.0

        # scale == 0 => noise_final, no noise
        noise_scale = self.noise_final * (1 - scale) + self.noise_initial * scale

        self.log("Train/noise_scale", noise_scale)

        if scale > 1e-6:  # If the scale is significant, we add noise.
            noise = torch.randn_like(x) * noise_scale * 0.04  # [b, n, 3]
            bbox = x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]  # [b, 1, 3]
            noise = noise * bbox  # Scale the noise by the bounding box size.
            x = x + noise  # Add noise to the input point cloud.

        # noise_scale==0 => no noise, no masking
        n_points_remove = roundup_16(int(noise_scale * x.shape[1] * self.masking_dropout_p_upper))
        if self.global_step < 5:
            n_points_remove = 0

        if n_points_remove > 0:
            perm = torch.randperm(x.shape[1], device=x.device)[:n_points_remove]
            mass[:, perm, :] *= 0.01 # filter out the points, as masking.
            mass = mass / mass.mean(dim=1, keepdim=True)

        return x, y, mass  # Return the noisy input, target, and mass tensor.

    def forward(self, x, mass):
        # TODO: Use func_dim=0 in this model. mass as input is not sound technically.
        y, reg_tok = self.model(x, x, mass, return_register_tokens=True)  # b, n, c

        # Orthogonalize the output based on the selected algorithm.
        with torch.autocast(device_type="cuda", enabled=False):
            return qr_orthogonalization(y, mass), y, reg_tok  # Use QR orthogonalization.

    def _extract_batch(self, batch):
        x = batch["points"]
        mass = batch["mass"]
        y = batch['evecs']
        return x, y, mass

    def training_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        with torch.autocast(device_type=x.device.type, enabled=False):
            rot = random_rotate_3d_batched(self.random_rotate, x.shape[0])
            x2 = torch.bmm(x, torch.tensor(rot, device=x.device, dtype=x.dtype))
            x2 = x2 / torch.max(torch.abs(x2).view(x.shape[0], -1), dim=1, keepdim=True).values.unsqueeze(-1)

        with torch.no_grad():
            _, teacher_reg_tok = self.teacher(x2, x2, mass, return_register_tokens=True)

        teacher_reg_tok = self.head_dino(teacher_reg_tok[:, 0, :])
        with torch.autocast(device_type=x.device.type, enabled=False):
            C_batch = teacher_reg_tok.mean(dim=0).detach() # (b, c) -> (c,)
            new_C = self.momentum_C * self.teacher_C + (1 - self.momentum_C) * C_batch
            if self.trainer.world_size > 1:
                dist.all_reduce(new_C, op=dist.ReduceOp.AVG)
            self.teacher_C.copy_(new_C)
        teacher_reg_tok = teacher_reg_tok - self.teacher_C.view(1, -1) # (b, c) - (1, c)
        teacher_reg_tok = teacher_reg_tok / self.teacher_temperature
        self.log("Train/teacher_C", self.teacher_C.abs().mean().item(), prog_bar=True)

        x, y, mass = self._add_noise(x, y, mass)
        y_hat, y_original, stu_reg_tok = self(x, mass)
        stu_reg_tok = stu_reg_tok[:, 0, :] / self.student_temperature
        total: torch.Tensor = 0  # type: ignore

        with torch.autocast(device_type=x.device.type, enabled=False):  # use float32 for more precise computation.
            y_hat = y_hat.to(torch.float32)
            for loss_fn, weight in zip(self.losses, self.weights):
                # TODO: Metrics may accept `mask`; removed here since masking is deprecated.
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                if weight > 0:
                    total = total + loss_value * weight
                self.log(f"Train/{type(loss_fn).__name__}", loss_value, prog_bar=True)
            div = kl_div(stu_reg_tok, teacher_reg_tok)
            # Self-Supervised Knowledge Distillation
            total = total + div * self.div_weight

            # Avoid degenerate case where all batch are the same:
            koleo = self.koleo(stu_reg_tok)
            total = total + koleo * self.koleo_weight

            self.log("Train/Div", div, prog_bar=True)
            self.log("Train/KoLeo", koleo, prog_bar=True)
            self.log("Train/total", total, prog_bar=False)

        self.log("Train/mesh_size", x.shape[1])
        return total

    def validation_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original, _ = self(x, mass)  # No masking during validation.
        with torch.no_grad():
            total: torch.Tensor = 0  # type: ignore
            for loss_fn, weight in zip(self.val_losses, self.val_weights):
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                total = total + loss_value * weight
                self.log(f"Val/{type(loss_fn).__name__}", loss_value, sync_dist=True)
            self.log("Val/total", total, prog_bar=True, on_step=False, sync_dist=True)


            # select some of the eigenvectors to compute orthogonality loss.
            idx = int(math.log(self.targ_dim, 2))  # log2(targ_dim) gives the number of eigenvectors to use.
            for i in range(0, idx + 1):
                evec_idx = (
                    0 if i == 0 else int(2 ** (i - 1))
                )  # Select the first eigenvector for i=0, then 2^(i-1) for others.
                evec = y[:, :, [evec_idx]]  # Select the eigenvector
                self.log(f"Val_evec/rmse_{i}", self.ispan(pred=y_hat, target=evec, mass=mass), sync_dist=True)

        if batch_idx == 0:
            # Try to solve the generalized eigenvalue problem (GEVP) for the first sample
            evec, evec_gt = solve_gev_from_subspace_with_gt(y_hat[0], x[0], mass[0], k=self.targ_dim)  # [n, c], [n, c]
            m0 = mass[0].detach().cpu().numpy()  # Convert mass to numpy for GEVP.
            # multiply by -1 does not hurt the eigenvectors.
            scores = np.abs(outer_cosine_similarity(evec, evec_gt, M=m0))  # (c, d)
            max_scores = np.max(np.abs(scores), axis=0)  # [d]

            self.log("Val/evec_max_scores_mean", 1 - max_scores.mean(), sync_dist=True)
            self.log("Val/evec_max_scores_max", 1 - max_scores.max(), sync_dist=True)
            self.log("Val/evec_max_scores_min", 1 - max_scores.min(), sync_dist=True)
            self.log("Val/evec_max_scores_median", 1 - np.median(max_scores), sync_dist=True)
            self.log("Val/evec_max_scores_mismatch", 1 - np.trace(np.abs(scores)) / scores.shape[1], sync_dist=True)

        return total

    def configure_optimizers(self):  # type: ignore
        # Use shared factory to create optimizer and scheduler from cfg
        normal, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('register') or name == 'output_norm':
                print(f"Add {name} to no_decay")
                no_decay.append(param)
            else:
                normal.append(param)

        optimizer = optim.AdamW(
            [
                {"params": normal, "weight_decay": self.optimizer_config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.optimizer_config.max_lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
        )

        warmup_ratio = self.scheduler_config.warmup

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.optimizer_config.max_lr,
            total_steps=self.epochs,
            pct_start=warmup_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
