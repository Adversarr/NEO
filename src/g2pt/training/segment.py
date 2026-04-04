from lightning import LightningModule
import torch
import torch.nn.functional as F

from g2pt.metrics.cross_entropy import CrossEntropyLossForSegmentation
from g2pt.neuralop.model import get_model
from g2pt.training.common import create_optimizer_and_scheduler, accuracy
from g2pt.training.common import load_partial_state_dict_strict

from torchmetrics.segmentation import MeanIoU

class SegmentTraining(LightningModule):
    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.n_classes: int = num_classes
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

        self.loss = CrossEntropyLossForSegmentation()

        self.head_segmentation = get_model(
            phys_dim=3,
            func_dim=cfg.targ_dim_model,
            out_dim=self.n_classes,
            config=cfg.seg_model,
        )
        self.head_segmentation.reset_parameters()
        self.miou = MeanIoU(num_classes=self.n_classes, input_format="index")
        self.train_miou = MeanIoU(num_classes=self.n_classes, input_format="index")

    def forward(self, x, mass: torch.Tensor) -> torch.Tensor:
        y = self.eig_model(x, x, mass)
        logits = self.head_segmentation(x, y, mass)
        return logits

    @staticmethod
    def _transfer_face(x, faces):
        # Remap to faces
        x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
        faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
        xf = torch.gather(x_gather, 1, faces_gather)
        x_out = torch.mean(xf, dim=-1)
        return x_out

    def _to_face_logits_and_mass(self, y_hat: torch.Tensor, mass: torch.Tensor, faces: torch.Tensor):
        if faces.ndim == 2:
            faces = faces.unsqueeze(0)
        y_faces = self._transfer_face(y_hat, faces)
        m_faces = self._transfer_face(mass, faces)
        return y_faces, m_faces

    def training_step(self, batch, batch_idx):
        x = batch["points"]
        y = batch["labels"]
        # mass = batch["mass"]
        # mass = mass / torch.mean(mass, dim=1, keepdim=True)
        mass = torch.ones_like(batch['mass'])
        y_hat = self(x, mass)
        if "faces" in batch and "face_labels" in batch:
            faces = batch["faces"].long()
            y_hat, mass = self._to_face_logits_and_mass(y_hat, mass, faces)
            y = batch["face_labels"]
        cw = batch.get("class_weights", None)
        if cw is not None and cw.ndim > 1:
            cw = cw[0]
        loss = self.loss(y_hat, y, class_weights=cw)
        self.log("Train/Loss", loss, prog_bar=True)
        acc = accuracy(y_hat, y)
        self.log("Train/Acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        pred_idx = y_hat.argmax(dim=-1)
        self.log("Train/MIoU", self.train_miou(pred_idx, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["points"]
        y = batch["labels"]
        # mass = batch["mass"]
        # mass = mass / torch.mean(mass, dim=1, keepdim=True)
        mass = torch.ones_like(batch['mass'])
        y_hat = self(x, mass)
        if "faces" in batch and "face_labels" in batch:
            faces = batch["faces"].long()
            y_hat, mass = self._to_face_logits_and_mass(y_hat, mass, faces)
            y = batch["face_labels"]
        cw = batch.get("class_weights", None)
        if cw is not None and cw.ndim > 1:
            cw = cw[0]
        loss = self.loss(y_hat, y, class_weights=cw)
        self.log("Val/Loss", loss, sync_dist=True, prog_bar=True)
        acc = accuracy(y_hat, y)
        self.log("Val/Acc", acc, sync_dist=True, prog_bar=True)
        pred_idx = y_hat.argmax(dim=-1)
        self.log("Val/MIoU", self.miou(pred_idx, y), on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["points"]
        y = batch["labels"]
        mass = batch["mass"]
        mass = mass / torch.mean(mass, dim=1, keepdim=True)
        y_hat = self(x, mass)
        if "faces" in batch and "face_labels" in batch:
            faces = batch["faces"].long()
            y_hat, mass = self._to_face_logits_and_mass(y_hat, mass, faces)
            y = batch["face_labels"]
        cw = batch.get("class_weights", None)
        if cw is not None and cw.ndim > 1:
            cw = cw[0]
        loss = self.loss(y_hat, y, mass, class_weights=cw)
        self.log("Test/Loss", loss, sync_dist=True, prog_bar=True)
        acc = accuracy(y_hat, y)
        self.log("Test/Acc", acc, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        # Use shared factory to create optimizer and scheduler from cfg
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.head_segmentation if self.freeze_pretrained else self,
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
