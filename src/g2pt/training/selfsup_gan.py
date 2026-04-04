from torch.nn import ModuleList
from g2pt.metrics import get_metric
from g2pt.metrics.selfsupervised import SelfSupervisedLoss
from g2pt.optim.muon import Muon, build_muon_param_groups
from g2pt.utils.ortho_operations import qr_orthogonalization
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
import numpy as np
import torch
from torch import distributed as dist, optim
from copy import deepcopy
from g2pt.metrics.span import ProjectionLoss, SelfDistance
from g2pt.neuralop.model import get_model, get_sol_model
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_with_gt
from g2pt.training.common import create_optimizer_and_scheduler, load_partial_state_dict_strict, load_params_state_dict_strict

##### Model #####

class SelfSupervisedGANTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.automatic_optimization = False
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
        self.accumulate_steps = cfg.accumulate_grad_batches

        self.loss = SelfSupervisedLoss()
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

        # Pretraining Stage
        self.losses = ModuleList()
        self.weights = []
        for metric_args in cfg.training_metrics:
            kwargs = metric_args.get("kwargs", {})
            self.losses.append(get_metric(name=metric_args["name"], **kwargs))
            self.weights.append(metric_args.get("weight", 1.0))

    def forward(self, x: torch.Tensor, mass: torch.Tensor, rhs: torch.Tensor, is_training_eig: bool = True, q_basis: torch.Tensor | None = None):
        # rhs: (b, n, nsamples)
        if q_basis is None:
            space_vecs = self.model(x, fx=x, mass=mass)  # (b, n, c)
            # if self.enable_ewa_eig_model:
            #     with torch.no_grad():
            #         ewa_space_vecs = self.ewa_eig_model(x, fx=x, mass=mass)  # (b, n, c)
            #     space_vecs = ewa_space_vecs + (space_vecs - space_vecs.detach())
            # option 1: norm
            vol = torch.sum(mass, dim=1, keepdim=True)  # (b, 1, 1)
            space_vecs_norm = torch.sum(space_vecs.square() * mass, dim=1, keepdim=True) / vol  # (b, 1, c)
            q_basis = space_vecs / (torch.sqrt(space_vecs_norm) + 1e-6)  # (b, n, c)

        if not is_training_eig: # Detach q_basis if not training eig.
            q_basis = q_basis.detach()

        # Solve the system
        coeff, trial_basis = self.sol_model(x, qx=x, kvx=q_basis, rhs=rhs, mass_q=mass, mass_kv=mass)

        if is_training_eig: # Disable gradient to sol_model.
            coeff = coeff.detach()

        y_for_solver = torch.bmm(q_basis, coeff)  # (b,n,c), (b,c,nrhs) -> (b,n,nrhs)
        return y_for_solver, q_basis, trial_basis

    def sample_b(self, mass: torch.Tensor):
        """Sample b random right-hand sides per shape (b in demo)."""
        b, n, _ = mass.shape
        b_unbiased = torch.randn((b, n, self.n_samples), dtype=mass.dtype, device=mass.device) # (b, n, self.n_samples)
        # Cov(b) = M^{-1}
        b = b_unbiased / torch.clamp_min(torch.sqrt(mass), 1e-5)
        return b

    def training_step(self, batch, batch_idx):
        opt_eig, opt_sol = self.optimizers()  # type: ignore

        # 1. Train eig model
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        q_basis = None
        if not self.freeze_pretrained:
            b = self.sample_b(mass)
            x_hat, q_basis, trial_basis = self(x, mass, b, is_training_eig=True) # (b,n,1), (b,n,c), (b,n,c)

            # For pretrain part
            pretrain_x = batch['pretrain_points']
            pretrain_y = batch['pretrain_evecs']
            pretrian_mass = batch['pretrain_mass']
            pretrain_y_original = self.model(pretrain_x, fx=pretrain_x, mass=pretrian_mass)
            pretrain_y_hat = qr_orthogonalization(pretrain_y_original, pretrian_mass)

            with torch.autocast(x.device.type, enabled=False):
                # 1. Self-supervised
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
                total_loss = self_supervised

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
                    self.log(f"Train/PRE-{type(loss_fn).__name__}", loss_value)

                self.log("Train/mesh_size", x.shape[1])
                self.log("Train/loss", total_loss, prog_bar=True)
                self.log("Train/self_supervise", self_supervised, prog_bar=True)
                self.log("Train/sol_model_scale", self.sol_model.output_norm.data)

                if batch_idx == 0 and self.global_rank == 0 and self.current_epoch > 0:
                    log_dict = {}
                    try:
                        ranks = torch.linalg.matrix_rank(q_basis)
                        log_dict["Train/rank"] = ranks.float().mean().item()
                    except Exception as e:
                        self.print(f"Warning(train): Rank validation failed: {e}")
                    try :
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

            opt_eig.zero_grad()
            self.manual_backward(total_loss / self.accumulate_steps)
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.clip_gradients(opt_eig, gradient_clip_val=0.1, gradient_clip_algorithm="norm")
                opt_eig.step()

            q_basis = q_basis.detach()

        # 2. Train sol model
        opt_sol.zero_grad()
        b = self.sample_b(mass)
        # Reuse q_basis from eig training to save computation if needed, but recomputing is safer for graph detachment
        # However, since we detach in forward when is_training_eig=False, it is fine.
        # But wait, if we recompute, we run model again. To optimize, we should pass detached q_basis.
        x_hat, q_basis, trial_basis = self(x, mass, b, is_training_eig=False, q_basis=q_basis) # (b,n,1), (b,n,c), (b,n,c)
        with torch.autocast(x.device.type, enabled=False):
            self_supervised = self.loss(
                x_hat=x_hat,
                rhs=b,
                sysmat=(sysmat_indices, sysmat_values),
                subspace_vectors=q_basis,
                mass=mass,
                sysmat_csr=sysmat_csr,
            )
        self.manual_backward(self_supervised / self.accumulate_steps)
        if (batch_idx + 1) % self.accumulate_steps == 0:
            self.clip_gradients(opt_sol, gradient_clip_val=1, gradient_clip_algorithm="norm")
            opt_sol.step()

        return

    def on_train_epoch_end(self) -> None:
        sch_eig, sch_sol = self.lr_schedulers()  # type: ignore
        sch_eig.step()
        sch_sol.step()

    def validation_step(self, batch, batch_idx):
        x = batch['points']
        mass = batch["lumped_mass"]  # for physics
        b = self.sample_b(mass)
        x_hat, q_basis, trial_basis = self(x, mass, b)  # (b,n,nsamples), (b,n,c), (b,n,c)

        # For pretrain part
        pretrain_x = batch['pretrain_points']
        pretrain_y = batch['pretrain_evecs']
        pretrian_mass = batch['pretrain_mass']
        pretrain_y_original = self.model(pretrain_x, fx=pretrain_x, mass=pretrian_mass)
        pretrain_y_hat = qr_orthogonalization(pretrain_y_original, pretrian_mass)

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
            total_loss = self_supervised

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
                self.log(f"Val/PRE-{type(loss_fn).__name__}", loss_value)

        self.log("Val/total", total_loss)
        self.log("Val/self_supervise", self_supervised)

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
        # Lower LR for pretrained model
        optimizer = Muon(
            params=build_muon_param_groups(self.model),
            lr=self.optimizer_config.max_lr * 0.1 if not self.freeze_pretrained else 0,
            adamw_betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.optimizer_config.max_lr * 0.1 if not self.freeze_pretrained else 0,
            total_steps=self.epochs, # type: ignore
            pct_start=0.2,
            base_momentum=0.9,
            max_momentum=0.95,
            cycle_momentum=False,
        )

        # Optimizer 2:
        optimizer2 = Muon(
            params=build_muon_param_groups(self.sol_model),
            lr=self.optimizer_config.max_lr,
            adamw_betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
        )
        scheduler2 = optim.lr_scheduler.OneCycleLR(
            optimizer2,
            max_lr=self.optimizer_config.max_lr,
            total_steps=self.epochs, # type: ignore
            pct_start=0.2,
            base_momentum=0.9,
            max_momentum=0.95,
            cycle_momentum=False,
        )
        return [optimizer, optimizer2], [scheduler, scheduler2]

    def extract_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "sol_model": self.sol_model.state_dict(),
        }