import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

# Add src to path to ensure we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from g2pt.data.datasets.smplx_deform import SMPLXDeformationDataset, collate_fn
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.smpl_deform import SMPLDeformTraining


class SMPLXDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SMPLXDeformationDataset(
                data_file=self.cfg.train_file,
                enable_rotate=self.cfg.enable_rotate,
                target_k=self.cfg.target_k,
                delta=self.cfg.delta,
            )
            
            # Check if val_file exists
            if Path(self.cfg.val_file).exists():
                self.val_dataset = SMPLXDeformationDataset(
                    data_file=self.cfg.val_file,
                    enable_rotate=0.0, # No rotation for validation
                    target_k=self.cfg.target_k,
                    delta=self.cfg.delta,
                )
            else:
                print(f"Warning: Validation file {self.cfg.val_file} not found. Skipping validation dataset.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        return None


@hydra.main(version_base=None, config_path="../conf", config_name="smpl_deform")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    print(f"Experimental Name: {cfg.experiment_name}")

    # Initialize DataModule
    datamod = SMPLXDataModule(cfg.datamod)
    datamod.setup()
    
    # Determine pose_dim dynamically if possible from the dataset
    if datamod.train_dataset is not None:
        # Load one sample to check pose dimension
        sample = datamod.train_dataset[0]
        if "poses" in sample:
            pose_dim = sample["poses"].shape[0]
            print(f"Detected pose_dim from dataset: {pose_dim}")
            # Update cfg.pose_dim. 
            # Note: OmegaConf config objects might be frozen, so we unlock or just pass it to model
            cfg.pose_dim = pose_dim
        else:
            print(f"Warning: 'poses' not found in dataset. Using config default: {cfg.pose_dim}")

    # Initialize Model
    model = SMPLDeformTraining(cfg)

    # Logger
    logger = MLFlowLogger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)

    # Callbacks
    callbacks = [
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            save_top_k=2,
            save_last=True,
            monitor="Val/loss" if datamod.val_dataset else "train_loss",
            mode="min",
            auto_insert_metric_name=False,
            filename="{epoch}-{step}-{val_loss:.4f}" if datamod.val_dataset else "{epoch}-{step}-{train_loss:.4f}",
        ),
        GradNormMonitor(interval=100),
    ]

    # Profiler
    profiler = None
    if cfg.profile:
        from lightning.pytorch.profilers import PyTorchProfiler
        from torch.profiler import ProfilerActivity
        
        Path("./profile").mkdir(parents=True, exist_ok=True)
        profiler = PyTorchProfiler(
            dirpath="./profile",
            export_to_chrome=True,
            profile_memory=True,
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
        )

    # Strategy
    device_count = torch.cuda.device_count()
    if cfg.strategy == "ddp" or (cfg.strategy == "auto" and device_count > 1):
        strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    else:
        strategy = "auto"

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        strategy=strategy,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        gradient_clip_val=cfg.gradient_clip_val,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        overfit_batches=cfg.overfit_batches,
        profiler=profiler,
        enable_model_summary=False,
    )

    # Train
    trainer.fit(model=model, datamodule=datamod)


if __name__ == "__main__":
    main()
