from g2pt.metrics.selfsupervised import SelfSupervisedLoss
from g2pt.utils.ortho_operations import qr_orthogonalization
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
import numpy as np
import torch
from torch import distributed as dist
from copy import deepcopy
from g2pt.metrics.span import ProjectionLoss, SelfDistance
from g2pt.neuralop.model import get_model, get_sol_model
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_with_gt
from g2pt.training.common import create_optimizer_and_scheduler, load_partial_state_dict_strict, load_params_state_dict_strict

##### Model #####

class SelfSupervisedTraining(LightningModule):
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
            load_partial_state_dict_strict(self.model, ckpt_path=pretrained, key_prefix='model.')
            has_pretrain = True

        # Sol model
        self.sol_model = get_sol_model(3, 3, self.model.out_dim, cfg.sol_model)
        self.sol_model.reset_parameters()  # Reset parameters of the model.

        # Training
        epochs: int = cfg.get("epochs", 500)
        self.epochs = epochs

        # Read optimizer/scheduler from Hydra cfg (exp/conf/optim.yaml)
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler

        self.loss = SelfSupervisedLoss()
        self.regu = SelfDistance(asymmetric=True)
        self.weight_regu = cfg.get("weight_regu", 1e-3)
        self.n_samples = cfg.get("n_samples", 256)
        self.proj_loss = ProjectionLoss()

        if cfg.ckpt_full:
            print(f"🎉 Strict params-only resume from {cfg.ckpt_full}")
            load_params_state_dict_strict({
                'model.': self.model,
                'sol_model.': self.sol_model,
            }, ckpt_path=cfg.ckpt_full)
            has_pretrain = True

        if not has_pretrain:
            print("⚠️ No pretrained model params load!")

        if self.freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("🧊 Freezing pretrained model.")

        self.enable_ewa_eig_model = cfg.get("enable_ewa_eig_model", False)
        self.ewa_momentum = cfg.get("ewa_momentum", 0.999)
        if self.enable_ewa_eig_model:
            self.ewa_eig_model = deepcopy(self.model)
            for param in self.ewa_eig_model.parameters():
                param.requires_grad = False
            self.ewa_eig_model.eval()
            print("⛏️ Use EWA to train eig model.")

    def forward(self, x: torch.Tensor, mass: torch.Tensor, rhs: torch.Tensor):
        # rhs: (b, n, nsamples)
        space_vecs = self.model(x, fx=x, mass=mass)  # (b, n, c)
        if self.enable_ewa_eig_model:
            with torch.no_grad():
                ewa_space_vecs = self.ewa_eig_model(x, fx=x, mass=mass)  # (b, n, c)
            space_vecs = ewa_space_vecs + (space_vecs - space_vecs.detach())
        # option 1: norm
        vol = torch.sum(mass, dim=1, keepdim=True)  # (b, 1, 1)
        space_vecs_norm = torch.sum(space_vecs.square() * mass, dim=1, keepdim=True) / vol  # (b, 1, c)
        q_basis = space_vecs / (torch.sqrt(space_vecs_norm) + 1e-6)  # (b, n, c)

        # Solve the system
        coeff, trial_basis = self.sol_model(x, qx=x, kvx=q_basis, rhs=rhs, mass_q=mass, mass_kv=mass)

        y_for_solver = torch.bmm(q_basis, coeff)  # (b,n,c), (b,c,nrhs) -> (b,n,nrhs)
        return y_for_solver, q_basis, trial_basis

    def sample_b(self, mass: torch.Tensor):
        """Sample b random right-hand sides per shape (b in demo)."""
        b, n, _ = mass.shape
        b_unbiased = torch.randn((b, n, self.n_samples), dtype=mass.dtype, device=mass.device) # (b, n, self.n_samples)
        # Cov(b) = M^{-1}
        b = b_unbiased / torch.clamp_min(torch.sqrt(mass), 1e-5)
        return b

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.enable_ewa_eig_model:
            self._ema_update_eig_model()

    def _ema_update_eig_model(self) -> None:
        if self.global_rank == 0:
            for param, ewa_param in zip(self.model.parameters(), self.ewa_eig_model.parameters()):
                ewa_param.data.mul_(self.ewa_momentum).add_(param.data, alpha=1 - self.ewa_momentum)
            for ema_buf, model_buf in zip(self.ewa_eig_model.buffers(), self.model.buffers()):
                ema_buf.data.copy_(model_buf.data)
        if self.trainer.world_size > 1:
            for param in self.ewa_eig_model.parameters():
                dist.broadcast(param.data, src=0)
            for ema_buf in self.ewa_eig_model.buffers():
                dist.broadcast(ema_buf.data, src=0)
        self.ewa_eig_model.eval()

    def training_step(self, batch, batch_idx):
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        b = self.sample_b(mass)
        x_hat, q_basis, trial_basis = self(x, mass, b) # (b,n,1), (b,n,c), (b,n,c)

        with torch.autocast(x.device.type, enabled=False):
            sysmat_indices = batch['stiff_indices'].to(x.device, dtype=torch.long)
            sysmat_values = batch['stiff_values'].to(x.device, dtype=torch.float32)
            sysmat_csr = batch.get("stiff_csr", None)
            x_hat = x_hat.float()
            b = b.float()
            mass = mass.float()
            trial_basis = trial_basis.float()
            q_basis = q_basis.float()
            self_supervised = self.loss(
                x_hat=x_hat,
                rhs=b,
                sysmat=(sysmat_indices, sysmat_values),
                subspace_vectors=q_basis,
                mass=mass,
                sysmat_csr=sysmat_csr,
            )

            eigen_ortho = self.regu(pred=q_basis, mass=mass, y_original=q_basis)
            reg_loss = eigen_ortho
            total_loss = self_supervised + self.weight_regu * reg_loss

            self.log("Train/mesh_size", x.shape[1], prog_bar=True)
            self.log("Train/loss", total_loss, prog_bar=True)
            self.log("Train/self_supervise", self_supervised, on_step=True, prog_bar=False)
            self.log("Train/regularity", reg_loss, on_step=True, prog_bar=False)
            self.log("Train/sol_model_scale", self.sol_model.output_norm.data)

            if batch_idx == 0 and self.global_rank == 0 and self.current_epoch > 0:
                log_dict = {}
                try:
                    ranks = torch.linalg.matrix_rank(q_basis)
                    log_dict["Train/rank"] = ranks.float().mean().item()
                except Exception as e:
                    self.print(f"Warning(train): Rank validation failed: {e}")
                try:
                    q = qr_orthogonalization(q_basis, mass)
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
                self.log_dict(log_dict, rank_zero_only=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        b = self.sample_b(mass)
        x_hat, q_basis, trial_basis = self(x, mass, b)  # (b,n,nsamples), (b,n,c), (b,n,c)

        with torch.autocast(x.device.type, enabled=False):
            sysmat_indices = batch['stiff_indices'].to(x.device, dtype=torch.long)
            sysmat_values = batch['stiff_values'].to(x.device, dtype=torch.float32)
            sysmat_csr = batch.get("stiff_csr", None)
            # Use float32 for numerics in loss
            x_hat = x_hat.float()
            b = b.float()
            mass = mass.float()
            trial_basis = trial_basis.float()
            q_basis = q_basis.float()
            self_supervised = self.loss(
                x_hat=x_hat,
                rhs=b,
                sysmat=(sysmat_indices, sysmat_values),
                subspace_vectors=q_basis,
                mass=mass,
                sysmat_csr=sysmat_csr,
            )
            regularity = self.regu(pred=q_basis, mass=mass, y_original=q_basis)

        total_loss = self_supervised + self.weight_regu * regularity
        self.log("Val/total", total_loss)
        self.log("Val/self_supervise", self_supervised)
        self.log("Val/regularity", regularity)

        if batch_idx == 0 and self.current_epoch > 0 and self.global_rank == 0:
            log_dict = {}
            try:
                ranks = torch.linalg.matrix_rank(q_basis)
                log_dict["Val/rank"] = ranks.float().mean().item()
            except Exception as e:
                self.print(f"Warning(val): Rank validation failed: {e}")
            try:
                q = qr_orthogonalization(q_basis, mass)
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
            self.log_dict(log_dict, rank_zero_only=True)

        return total_loss

    def configure_optimizers(self):  # type: ignore
        # Use shared factory to create optimizer and scheduler from cfg
        module = self if self.freeze_pretrained else self.sol_model
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=module,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def extract_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "sol_model": self.sol_model.state_dict(),
        }