from lightning import LightningModule
import torch

from g2pt.neuralop.model import get_model
from g2pt.training.common import create_optimizer_and_scheduler, accuracy

from g2pt.training.common import load_partial_state_dict_strict
from torch import nn
from g2pt.utils.correspondence import compute_correspondence, compute_correspondence_batched


class CorrespondenceTraining(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.k_corr: int = cfg.datamod.k_corr  # correspondence matrix dimension.
        self.eig_model = get_model(phys_dim=3, func_dim=3, out_dim=cfg.targ_dim_model, config=cfg.model)
        self.epochs = cfg.epochs
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler
        pretrained = self.cfg.ckpt_pretrain
        freeze = self.cfg.freeze_pretrained
        self.freeze_pretrained = freeze
        if pretrained:
            print(f"🎉 Loading pretrained model (partial, strict) from {pretrained}.")
            load_partial_state_dict_strict(self.eig_model, ckpt_path=pretrained, key_prefix='model.')
            if freeze:
                print("🧊 Freezing pretrained model.")
                for param in self.eig_model.parameters():
                    param.requires_grad = False
                self.eig_model.eval()
        else:
            print("⚠️ No pretrained model is loaded.")

        self.loss = nn.HuberLoss()

        self.head_correspondence = get_model(
            phys_dim=3,
            func_dim=cfg.targ_dim_model,
            out_dim=self.k_corr,
            config=cfg.corr_model,
        )
        self.head_correspondence.reset_parameters()

    def forward(self, x, mass):
        y = self.eig_model(x, x, mass)  # b, n, c
        x = self.head_correspondence(x, y, mass)
        return x

    def compute_correspondence(self, batch):
        p1, m1 = batch["p1"], batch["m1"]
        p2, m2 = batch["p2"], batch["m2"]
        feat1 = self(p1, m1)     # (b, n, k)
        feat2 = self(p2, m2)     # (b, n, k)
        eval1 = batch["evals1"]  # (b, k)
        eval2 = batch["evals2"]  # (b, k)
        evec1 = m1 * batch["evecs1"]  # (b, n, k)
        evec2 = m2 * batch["evecs2"]  # (b, n, k)
        C_pred = compute_correspondence(feat1, feat2, eval1, eval2, evec1.mT, evec2.mT)
        return C_pred, feat1, feat2

    def training_step(self, batch, batch_idx):
        C_gt = batch["C_gt"]  # (b, k, k)
        C_pred, *_ = self.compute_correspondence(batch)

        loss = self.loss(C_pred, C_gt)
        self.log("Train/Loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        C_gt = batch["C_gt"]  # (b, k, k)
        C_pred, *_ = self.compute_correspondence(batch)
        loss = self.loss(C_pred, C_gt)
        self.log("Val/Loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        # Use shared factory to create optimizer and scheduler from cfg
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.head_correspondence if not self.freeze_pretrained else self,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=self.cfg.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }