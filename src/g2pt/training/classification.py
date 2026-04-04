from lightning import LightningModule
import torch

from g2pt.metrics.cross_entropy import CrossEntropyLossForClassification
from g2pt.neuralop.model import Transolver2Model, get_model
from g2pt.training.common import create_optimizer_and_scheduler, accuracy

from g2pt.training.common import load_partial_state_dict_strict
from g2pt.training.pretrain import PretrainTraining
from torch import nn

def _make_mlp_out(in_dim, out_dim):
    hidden = in_dim + out_dim
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )

class ClassificationTraining(LightningModule):
    def __init__(self, cfg, num_classes: int | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.n_classes: int = num_classes or cfg.datamod.n_class
        self.eig_model = get_model(phys_dim=3, func_dim=3, out_dim=cfg.targ_dim_model, config=cfg.model)
        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler
        pretrained = self.cfg.ckpt_pretrain
        freeze = self.cfg.freeze_pretrained
        self.freeze_pretrained = freeze
        if pretrained:
            print(f"🎉 Loading pretrained model (partial, strict) from {pretrained}.")
            # load_partial_state_dict_strict(self.eig_model, ckpt_path=pretrained, key_prefix='model.')
            ckpt = PretrainTraining.load_from_checkpoint(pretrained, weights_only=False, strict=False)
            self.eig_model = ckpt.model

            if freeze:
                print("🧊 Freezing pretrained model.")
                for param in self.eig_model.parameters():
                    param.requires_grad = False
                self.eig_model.eval()
        else:
            print("⚠️ No pretrained model is loaded.")

        self.loss = CrossEntropyLossForClassification()
        self.use_transolver2_model_head = cfg.use_transolver2_model_head
        if self.use_transolver2_model_head and isinstance(self.eig_model, Transolver2Model):
            out_dim = self.eig_model.out_dim
            self.head_cls = _make_mlp_out(out_dim, self.n_classes)

        else:
            self.head_cls = get_model(
                phys_dim=3,
                func_dim=cfg.targ_dim_model,
                out_dim=self.n_classes,
                config=cfg.cls_model,
            )
            self.head_cls.reset_parameters()


    def forward(self, x: torch.Tensor, mass: torch.Tensor):
        if self.use_transolver2_model_head and isinstance(self.eig_model, Transolver2Model):
            _, cls_tokens = self.eig_model(x, x, mass, return_register_tokens=True)
            logits = self.head_cls(cls_tokens[..., 0, :]) # (B, C)
        else:
            y = self.eig_model(x, x, mass)
            logits = self.head_cls(x, y, mass) # (B, N, C)
            logits = torch.mean(logits, dim=1) # (B, C)
        return logits # (B, C)

    def training_step(self, batch, batch_idx):
        x = batch['points']
        y = batch['label']
        # mass = batch['mass']
        # mass = mass / torch.mean(mass, dim=1, keepdim=True)
        mass = torch.ones_like(batch['mass'])
        logits = self(x, mass)
        cw = batch.get("class_weights", None)
        if cw is not None and cw.ndim > 1:
            cw = cw[0]
        loss = self.loss(logits, y, class_weights=cw)
        acc = accuracy(logits, y)
        mesh_sizes = x.shape[1]
        self.log("Train/MeshSize", mesh_sizes, on_step=False, on_epoch=True)
        self.log("Train/Loss", loss, on_step=True, prog_bar=True)
        self.log("Train/Acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['points']
        y = batch['label']
        # mass = batch['mass']
        # mass = mass / torch.mean(mass, dim=1, keepdim=True)
        mass = torch.ones_like(batch['mass'])
        logits = self(x, mass)
        cw = batch.get("class_weights", None)
        if cw is not None and cw.ndim > 1:
            cw = cw[0]
        loss = self.loss(logits, y, class_weights=cw)
        acc = accuracy(logits, y)
        self.log("Val/Loss", loss, on_epoch=True, prog_bar=True)
        self.log("Val/Acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        # Use shared factory to create optimizer and scheduler from cfg
        module = self.head_cls if not self.freeze_pretrained else self
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=module,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=self.cfg.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
