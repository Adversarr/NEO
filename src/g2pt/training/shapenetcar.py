from pathlib import Path
from lightning import LightningModule
import h5py
from torch.nn import functional as F
from g2pt.metrics.rrmse import RootRelMSELoss
from g2pt.neuralop.model import Transolver2SolverModel, TransolverNeXtSolverModel, get_model, get_sol_model
from g2pt.training.common import create_optimizer_and_scheduler, load_partial_state_dict_strict

class ShapenetCarSimulationTraining(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.surf_eig_model = get_model(phys_dim=3, func_dim=3, out_dim=cfg.targ_dim_model, config=cfg.model)
        self.epochs = cfg.epochs
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler
        pretrained = self.cfg.ckpt_pretrain
        freeze = self.cfg.freeze_pretrained
        self.freeze_pretrained = freeze
        if pretrained:
            print(f"🎉 Loading pretrained model (partial, strict) from {pretrained}.")
            load_partial_state_dict_strict(self.surf_eig_model, ckpt_path=pretrained, key_prefix='model.')
            if freeze:
                print("🧊 Freezing pretrained model.")
                for param in self.surf_eig_model.parameters():
                    param.requires_grad = False
                self.surf_eig_model.eval()
        else:
            print("⚠️ No pretrained model is loaded.")

        self.loss = RootRelMSELoss()

        q_dim = 4
        out_dim = 4
        self.sim_model = TransolverNeXtSolverModel(3, q_dim, self.surf_eig_model.out_dim, cfg.sol_model, out_dim=out_dim)

    def forward(self, surf_pos, surf_mass, full_pos, full_feat):
        # 1. Infer the surface features.
        surf_feat = self.surf_eig_model(surf_pos, surf_pos, surf_mass)  # b, n, c

        # 2. Infer the simulation model.
        _, y = self.sim_model(
            x=full_pos,
            qx=full_feat,
            kvx=surf_feat,
            x_kv=surf_pos,
            mass_q=None,
            mass_kv=surf_mass,
            return_pq_project=False,
        )

        return y

    def training_step(self, batch, batch_idx):
        y = batch['out']
        assert y.shape[0] == 1, "Only support batch size 1 for now."
        surf = batch['surf'].flatten()
        surf_pos, surf_mass = batch['surf_pos'], batch['surf_mass']
        full_pos, full_feat = batch['full_pos'], batch['full_feat']
        y_pred = self(surf_pos, surf_mass, full_pos, full_feat)

        y_mean = batch['y_mean'].unsqueeze(1) # (b, 1, 4)
        y_std = batch['y_std'].unsqueeze(1) # (b, 1, 4)
        y_pred = y_pred * y_std + y_mean # denormalize here
        flat_surf = surf.flatten()
        pred_pressure = y_pred[..., -1].flatten()[flat_surf].view(1, -1, 1)
        true_pressure = y[..., -1].flatten()[flat_surf].view(1, -1, 1)
        pred_velocity = y_pred[..., :-1]
        true_velocity = y[..., :-1]

        loss_pressure = self.loss(pred_pressure, true_pressure)
        loss_velocity = self.loss(pred_velocity, true_velocity)
        reg = 1
        total = loss_velocity + reg * loss_pressure
        self.log("Train/Loss", total, prog_bar=True)
        self.log("Train/velocity", loss_velocity, prog_bar=True)
        self.log("Train/pressure", loss_pressure, prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        y = batch['out']
        assert y.shape[0] == 1, "Only support batch size 1 for now."
        surf = batch['surf'].flatten()
        surf_pos, surf_mass = batch['surf_pos'], batch['surf_mass']
        full_pos, full_feat = batch['full_pos'], batch['full_feat']
        y_pred = self(surf_pos, surf_mass, full_pos, full_feat)
        y_mean = batch['y_mean'].unsqueeze(1)
        y_std = batch['y_std'].unsqueeze(1)
        y_pred = y_pred * y_std + y_mean
        flat_surf = surf.flatten()
        pred_pressure = y_pred[..., -1].flatten()[flat_surf].view(1, -1, 1)
        true_pressure = y[..., -1].flatten()[flat_surf].view(1, -1, 1)
        pred_velocity = y_pred[..., :-1]
        true_velocity = y[..., :-1]

        loss_pressure = self.loss(pred_pressure, true_pressure)
        loss_velocity = self.loss(pred_velocity, true_velocity)
        reg = 1
        total = loss_velocity + reg * loss_pressure
        self.log("Val/Loss", total)
        self.log("Val/velocity", loss_velocity)
        self.log("Val/pressure", loss_pressure, prog_bar=True)
        return total

    def configure_optimizers(self):
        o, s = create_optimizer_and_scheduler(
            self if not self.freeze_pretrained else self.sim_model,
            self.optimizer_config,
            self.scheduler_config,
            self.epochs,
        )
        return {
            'optimizer': o,
            'lr_scheduler': s
        }
