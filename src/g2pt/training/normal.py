from lightning import LightningModule
import torch
import torch.nn.functional as F

from g2pt.neuralop.model import get_model
from g2pt.training.common import create_optimizer_and_scheduler
from g2pt.training.common import load_partial_state_dict_strict


class NormalTraining(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Backbone model (e.g. Transolver)
        self.eig_model = get_model(phys_dim=3, func_dim=3, out_dim=cfg.targ_dim_model, config=cfg.model)
        
        # Normal estimation head
        self.head_normal = get_model(
            phys_dim=3,
            func_dim=cfg.targ_dim_model,
            out_dim=3, # Output is 3D normal vector
            config=cfg.nrm_model,
        )
        self.head_normal.reset_parameters()

        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler
        self.use_abs_diff = cfg.use_abs_diff
        
        pretrained = self.cfg.get("ckpt_pretrain", None)
        freeze = self.cfg.get("freeze_pretrained", False)
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

    def forward(self, x, mass: torch.Tensor) -> torch.Tensor:
        y = self.eig_model(x, x, mass)
        normals = self.head_normal(x, y, mass)
        # Normalize to unit vectors, ensure float
        with torch.autocast(x.device.type, enabled=False):
            normals = normals.to(torch.float32)
            normals = F.normalize(normals, p=2, dim=-1)
        return normals

    @torch.autocast(device_type="cuda", enabled=False)
    def _compute_loss(self, pred_normals, gt_normals):
        """
        Compute cosine similarity loss for normals.
        """
        cos_sim = torch.sum(pred_normals.float() * gt_normals.float(), dim=-1)
        
        if self.use_abs_diff:
            # Loss = 1 - mean(abs(dot(pred, gt)))
            loss = 1.0 - torch.abs(cos_sim).mean()
        else:
            # Loss = 1 - mean(dot(pred, gt))
            loss = 1.0 - cos_sim.mean()
        return loss

    def training_step(self, batch, batch_idx):
        x = batch["points"]
        y = batch["normals"]
        mass = batch["mass"]
        mass = mass / torch.mean(mass, dim=1, keepdim=True)
        
        y_hat = self(x, mass)
        loss = self._compute_loss(y_hat, y)
        
        self.log("Train/Loss", loss, prog_bar=True)
        
        # Angular error in degrees
        with torch.no_grad():
            cos_sim = torch.clamp(torch.sum(y_hat * y, dim=-1), -1.0, 1.0)
            angle_err = torch.acos(torch.abs(cos_sim)) * (180.0 / 3.14159265)
            self.log("Train/AngleErr", angle_err.mean(), prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["points"]
        y = batch["normals"]
        mass = batch["mass"]
        mass = mass / torch.mean(mass, dim=1, keepdim=True)
        
        y_hat = self(x, mass)
        loss = self._compute_loss(y_hat, y)
        
        self.log("Val/Loss", loss, sync_dist=True, prog_bar=True)
        
        with torch.no_grad():
            cos_sim = torch.clamp(torch.sum(y_hat * y, dim=-1), -1.0, 1.0)
            angle_err = torch.acos(torch.abs(cos_sim)) * (180.0 / 3.14159265)
            self.log("Val/AngleErr", angle_err.mean(), sync_dist=True, prog_bar=True)
            
        return loss

    def configure_optimizers(self):
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.head_normal if self.freeze_pretrained else self,
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
