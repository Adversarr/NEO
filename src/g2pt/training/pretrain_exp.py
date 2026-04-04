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
from g2pt.neuralop.model.transolver_experimental import TransolverExpModel
from g2pt.utils.gev import solve_gev_from_subspace_with_gt, outer_cosine_similarity
from g2pt.utils.ortho_operations import gram_schmidt_orthogonalization, newton_schulz, qr_orthogonalization
from g2pt.utils.common import roundup
from g2pt.training.common import create_optimizer_and_scheduler

class PretrainTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.targ_dim: int = cfg.get("targ_dim", 16)
        self.targ_dim_model: int = cfg.get("targ_dim_model", self.targ_dim)
        self.weight_decay: float = cfg.get("weight_decay", 0.01)
        self.ortho_algorithm = cfg.get("ortho_algorithm", "qr")  # Use QR decomposition for orthogonalization.
        self.muon = cfg.get("muon", False)  # Use Muon Optimizer
        self.model = TransolverExpModel(3, 3, self.targ_dim_model, cfg.model)
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

        if cfg.get("compile", False):
            self.model = torch.compile(self.model)

    def forward(self, x, mass):
        # TODO: Use func_dim=0 in this model. mass as input is not sound technically.
        y, lbl = self.model(x, x, mass)  # b, n, c

        # Orthogonalize the output based on the selected algorithm.
        with torch.autocast(device_type="cuda", enabled=False):
            return qr_orthogonalization(y, mass), y, lbl  # Use QR orthogonalization.

    def _extract_batch(self, batch):
        x = batch["points"]
        mass = batch["mass"]
        y = batch['evecs']
        return x, y, mass

    def training_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original, lbl = self(x, mass)
        total: torch.Tensor = 0.001 * lbl.mean()  # Add a load balancing term.

        with torch.autocast(device_type=x.device.type, enabled=False):  # use float32 for more precise computation.
            y_hat = y_hat.to(torch.float32)
            self.log("Train/LoadBalancing", lbl.mean(), prog_bar=True)
            for loss_fn, weight in zip(self.losses, self.weights):
                # TODO: Metrics may accept `mask`; removed here since masking is deprecated.
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                if weight > 0:
                    total = total + loss_value * weight
                self.log(f"Train/{type(loss_fn).__name__}", loss_value, prog_bar=True)
            self.log("Train/total", total, prog_bar=False)

        self.log("Train/mesh_size", x.shape[1], prog_bar=False)
        return total

    def validation_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original, lbl = self(x, mass)  # No masking during validation.

        total: torch.Tensor = 0.001 * lbl.mean()  # Add a load balancing term.
        self.log("Val/LoadBalancing", lbl.mean(), sync_dist=True)
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
            total_steps=self.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
