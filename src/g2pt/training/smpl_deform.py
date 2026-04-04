import torch
import numpy as np
from lightning import LightningModule
from g2pt.neuralop.model import get_model
from g2pt.training.common import create_optimizer_and_scheduler, load_partial_state_dict_strict

class SMPLDeformTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.lambda_v = cfg.get("lambda_v", 1.0)
        self.lambda_g = cfg.get("lambda_g", 1.0)
        self.pose_dim = cfg.get("pose_dim", 16) # Should match dataset pose dimension
        self.epochs = cfg.epochs
        
        # 1. Backbone Model (Pretrained usually)
        # It maps inputs (x) to some latent representation or features.
        # Following segment.py pattern:
        # We assume cfg.model defines the backbone.
        # targ_dim_model should be the output dimension of this backbone.
        self.backbone = get_model(3, 3, cfg.targ_dim_model, cfg.model)

        # Load pretrained backbone if specified
        pretrained = cfg.get("ckpt_pretrain", None)
        self.freeze_pretrained = cfg.get("freeze_pretrained", False)
        if pretrained:
            print(f"🎉 Loading pretrained backbone (partial, strict) from {pretrained}.")
            load_partial_state_dict_strict(self.backbone, ckpt_path=pretrained, key_prefix='model.')
            if self.freeze_pretrained:
                print("🧊 Freezing pretrained backbone.")
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()
        else:
            print("⚠️ No pretrained backbone is loaded.")

        # 2. Deformation Head / Adaptor
        # Takes backbone output + pose info -> Deformation (3D)
        # Input phys_dim=3 (original coords), func_dim=targ_dim_model (backbone features)
        # Note: In original single model version, we passed pose as func input.
        # Now we need to fuse pose into this head.
        # The user wants "output head/adaptor" similar to segment.py.
        # In segment.py: logits = self.head_segmentation(x, y, mass) where y is backbone output.
        # Here, we need to pass 'poses' as well.
        # The head model needs to handle `func_dim`.
        # If we follow segment.py, `head_segmentation` takes `y` (backbone out) as functional input `fx`.
        # But we also have `poses`.
        # We can concatenate `poses` to `backbone features` or `x`.
        # Let's assume the head expects:
        #   x: original points (3)
        #   fx: backbone features (targ_dim_model) + src_poses (pose_dim) + tar_poses (pose_dim)
        #   mass: mass
        
        # We concatenate both source and target poses
        head_func_dim = cfg.targ_dim_model + 2 * self.pose_dim
        
        self.head = get_model(
            phys_dim=3,
            func_dim=head_func_dim,
            out_dim=3, # Output is deformation vector (dx, dy, dz)
            config=cfg.dfm_model,
        )
        self.head.reset_parameters()
        
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler
        # Changed to 1.0 since data is already normalized in preprocessing
        self.std = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=False)

    def training_step(self, batch, batch_idx):
        points = batch["points"] # (B, N, 3)
        output_pos = batch["output_pos"] # (B, N, 3)
        src_poses = batch["src_poses"] # (B, D)
        tar_poses = batch["tar_poses"] # (B, D)
        mass = batch["lumped_mass"] # (B, N, 1)

        B, N, _ = points.shape
        
        # 1. Backbone Forward
        feats = self.backbone(points, points, mass=mass) # (B, N, targ_dim_model)
        feats_norm = torch.mean(feats ** 2, dim=-1, keepdim=True) # (B, N, 1)
        feats = feats / torch.sqrt(feats_norm + 1e-12) # Normalize features via RMS
        # 2. Prepare Head Input
        # Expand both poses to (B, N, D)
        src_poses_exp = src_poses.unsqueeze(1).expand(-1, N, -1)
        tar_poses_exp = tar_poses.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate backbone features with poses
        # fx for head = [feats, src_poses, tar_poses]
        head_fx = torch.cat([feats, src_poses_exp, tar_poses_exp], dim=-1) 
        
        # 3. Head Forward
        # Predicts deformation
        deformation = self.head(points, head_fx) * self.std.view(1, 1, 3) # (B, N, 3)
        
        # Predicted positions
        preds = points + deformation
        
        # --- Loss Computation (Force float32 for stability and cusparse compatibility) ---
        with torch.autocast('cuda', enabled=False):
            preds = preds.float()
            output_pos = output_pos.float()
            mass = mass.float()
            
            # Vertex Loss (L_vertex)
            # Formula: sum(m_i * ||v_hat - v_tar||^2)
            # Weighted MSE.
            diff_sq = torch.sum((preds - output_pos)**2, dim=-1, keepdim=True) # (B, N, 1)
            loss_v = (torch.sum(mass * diff_sq, dim=(1, 2)) / torch.sum(mass, dim=(1, 2))).mean()
            
            # Gradient Loss (L_gradient)
            # Approximated by LBO: || L * v_hat - L * v_tar ||^2
            if self.lambda_g > 0:
                # Use block-diagonal CSR matrix for efficiency
                stiff_csr = batch["stiff_csr"].to(points.device).float() # Ensure matrix is float32
                
                preds_flat = preds.reshape(-1, 3)
                output_pos_flat = output_pos.reshape(-1, 3)
                
                term_pred = torch.sparse.mm(stiff_csr, preds_flat).reshape(B, N, 3)
                term_target = torch.sparse.mm(stiff_csr, output_pos_flat).reshape(B, N, 3)
                
                h1_diff_sq = torch.sum((term_pred - term_target)**2, dim=-1, keepdim=True) # (B, N, 1)
                loss_g = (torch.sum(mass * h1_diff_sq, dim=(1, 2)) / torch.sum(mass, dim=(1, 2))).mean()
            else:
                loss_g = torch.tensor(0.0, device=points.device)

            # Total Loss
            loss = self.lambda_v * loss_v + self.lambda_g * loss_g
        
        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_v", loss_v, prog_bar=True)
        self.log("train/loss_g", loss_g, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        points = batch["points"]
        output_pos = batch["output_pos"]
        src_poses = batch["src_poses"]
        tar_poses = batch["tar_poses"]
        mass = batch["lumped_mass"]

        B, N, _ = points.shape
        
        feats = self.backbone(points, points, mass=mass)
        src_poses_exp = src_poses.unsqueeze(1).expand(-1, N, -1)
        tar_poses_exp = tar_poses.unsqueeze(1).expand(-1, N, -1)
        head_fx = torch.cat([feats, src_poses_exp, tar_poses_exp], dim=-1)
        deformation = self.head(points, head_fx, mass=mass) * self.std.view(1, 1, 3)
        preds = points + deformation
        
        # --- Loss Computation (Force float32 for stability and cusparse compatibility) ---
        with torch.autocast('cuda', enabled=False):
            preds = preds.float()
            output_pos = output_pos.float()
            mass = mass.float()
            
            diff_sq = torch.sum((preds - output_pos)**2, dim=-1, keepdim=True) # (B, N, 1)
            loss_v = (torch.sum(mass * diff_sq, dim=(1, 2)) / torch.sum(mass, dim=(1, 2))).mean()
            
            if self.lambda_g > 0:
                stiff_csr = batch["stiff_csr"].to(points.device).float()
                preds_flat = preds.reshape(-1, 3)
                output_pos_flat = output_pos.reshape(-1, 3)
                
                term_pred = torch.sparse.mm(stiff_csr, preds_flat).reshape(B, N, 3)
                term_target = torch.sparse.mm(stiff_csr, output_pos_flat).reshape(B, N, 3)
                h1_diff_sq = torch.sum((term_pred - term_target)**2, dim=-1, keepdim=True) # (B, N, 1)
                loss_g = (torch.sum(mass * h1_diff_sq, dim=(1, 2)) / torch.sum(mass, dim=(1, 2))).mean()
            else:
                loss_g = torch.tensor(0.0, device=points.device)

            loss = self.lambda_v * loss_v + self.lambda_g * loss_g
        
        self.log("Val/loss", loss, prog_bar=True)
        self.log("Val/loss_v", loss_v)
        self.log("Val/loss_g", loss_g)
        
        return loss

    def configure_optimizers(self):
        # We need to optimize both backbone and head, unless backbone is frozen.
        module = self.head if self.freeze_pretrained else self
        opt, lrs = create_optimizer_and_scheduler(
            module,
            self.optimizer_config,
            self.scheduler_config,
            self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lrs,
                "interval": "step",
            },
        }
