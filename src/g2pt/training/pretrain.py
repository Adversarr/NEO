import math
from typing import Tuple
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
import numpy as np
import torch
from torch.nn import ModuleList

from g2pt.metrics import get_metric
from g2pt.metrics.span import InverseSpanLoss, OrthogonalLoss
from g2pt.neuralop.model import get_model
from g2pt.utils.gev import solve_gev_from_subspace_with_gt, outer_cosine_similarity
from g2pt.utils.ortho_operations import gram_schmidt_orthogonalization, newton_schulz, qr_orthogonalization
from g2pt.utils.common import roundup
from g2pt.training.common import create_optimizer_and_scheduler

import time

class PretrainTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.targ_dim: int = cfg.get("targ_dim", 16)
        self.targ_dim_model: int = cfg.get("targ_dim_model", self.targ_dim)
        self.weight_decay: float = cfg.get("weight_decay", 0.01)
        self.ortho_algorithm = cfg.get("ortho_algorithm", "qr")  # Use QR decomposition for orthogonalization.
        self.muon = cfg.get("muon", False)  # Use Muon Optimizer
        self.model = get_model(3, 3, self.targ_dim_model, cfg.model)
        self.model.reset_parameters()  # Reset parameters of the model.

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
        for val_loss_args in cfg.get("validation_metrics", []):
            kwargs = val_loss_args.get("kwargs", {})
            self.val_losses.append(get_metric(name=val_loss_args["name"], **kwargs))
            self.val_weights.append(val_loss_args.get("weight", 1.0))

        self.losses = ModuleList()
        self.weights = []
        for metric_args in cfg.get("training_metrics", []):
            kwargs = metric_args.get("kwargs", {})
            self.losses.append(get_metric(name=metric_args["name"], **kwargs))
            self.weights.append(metric_args.get("weight", 1.0))

        # regularities:
        self.noise_initial = cfg.get("noise_initial", 0.02)
        self.noise_final = cfg.get("noise_final", 0.0)
        self.noise_ratio = cfg.get("noise_ratio", 0.3)
        self.noise_schedule = cfg.get("noise_schedule", "linear")  # linear or cosine schedule.

        if cfg.get("compile", False):
            self.model = torch.compile(self.model)

    def _add_noise(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [b, n, 3].
        """
        if not self.training or self.noise_schedule == "disable":  # During validation, we do not apply any masking.
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
            noise = torch.randn_like(x) * noise_scale  # [b, n, 3]
            bbox = x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]  # [b, 1, 3]
            noise = noise * bbox  # Scale the noise by the bounding box size.
            x = x + noise  # Add noise to the input point cloud.

        # 2. Mask some points in the input point cloud.
        lo, hi = self.masking_dropout_p_lower, self.masking_dropout_p_upper
        # dropout_p: float = (np.random.rand() * (hi - lo) + lo) * scale

        # 2.1 If the model is really a neural operator, the output should be invariant to the sampling rate of
        #     input point cloud. Randomly drop points by masking them in the input is a good way to examine and
        #     test the model's robustness.
        #     However, random droping of points will possibly lead to an incorrect mass matrix of the input point
        #     cloud. Recommend to use smaller dropout_p for the neural operator training for a better performance.
        #     At the very beginning, we allocate more GPU memory to guarantee the training will not break from OOM
        if self.masking_algorithm in ["dropout", "dropout_permute"] and self.global_step > 10:
            # implement dropout by keeping a fixed number of points (faster training)
            drop = (scale * hi + (1 - scale) * lo)  # dropout_p is a random value in [lo, hi]
            # Ensure we keep a reduced, multiple-of-256 count, capped by original N
            keep_est = int(roundup(int(x.shape[1] * (1 - drop)), 256))
            train_pc_count = min(max(keep_est, 256), x.shape[1])
            # TODO: Confirm the desired lower bound for train_pc_count; using 256 for kernel-friendliness.
            x = x[:, :train_pc_count, :]
            mass = mass[:, :train_pc_count, :]
            y = y[:, :train_pc_count, :]  # Adjust the target tensor to match the new input size.

        return x, y, mass  # Return the noisy input, target, and mass tensor.

    def forward(self, x, mass, no_mass=False, return_time=False):
        # TODO: Use func_dim=0 in this model. mass as input is not sound technically.
        start = time.perf_counter()
        if no_mass:
            y = self.model(x, x, None)
        else:
            y = self.model(x, x, mass)  # b, n, c
        if return_time:
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - start

        # Orthogonalize the output based on the selected algorithm.
        with torch.autocast(device_type="cuda", enabled=False):
            if self.training:
                perm = torch.randperm(y.shape[2], device=y.device)
                y = y[:, :, perm]

            if self.ortho_algorithm == "qr":
                if return_time:
                    return qr_orthogonalization(y, mass), y, infer_time
                else:
                    return qr_orthogonalization(y, mass), y
            elif self.ortho_algorithm == "gs":
                return gram_schmidt_orthogonalization(y, mass), y  # Orthogonalize the output.
            elif self.ortho_algorithm == "ns":
                return newton_schulz(y, mass), y  # Use Newton-Schulz iteration for orthogonalization.
            else:
                raise ValueError(f"Unknown orthogonalization algorithm: {self.ortho_algorithm}")

    def _extract_batch(self, batch):
        x = batch["points"]
        mass = batch["mass"]
        y = batch['evecs']
        return x, y, mass

    def training_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        x, y, mass = self._add_noise(x, y, mass)
        y_hat, y_original = self(x, mass)
        total: torch.Tensor = 0  # type: ignore

        with torch.autocast(device_type=x.device.type, enabled=False):  # use float32 for more precise computation.
            y_hat = y_hat.to(torch.float32)
            for loss_fn, weight in zip(self.losses, self.weights):
                # TODO: Metrics may accept `mask`; removed here since masking is deprecated.
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                if weight > 0:
                    total = total + loss_value * weight
                self.log(f"Train/{type(loss_fn).__name__}", loss_value, prog_bar=True)
            self.log("Train/total", total, prog_bar=False)

        self.log("Train/mesh_size", x.shape[1], prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original = self(x, mass)  # No masking during validation.
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
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.model,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
