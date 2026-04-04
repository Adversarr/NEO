from pathlib import Path

import hydra
from lightning import Trainer
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy
import torch

from g2pt.data.datasets.selfsup_experimental import SelfSupervisedExperimentalDataModule
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.selfsup_gan import SelfSupervisedGANTraining

@hydra.main(version_base=None, config_path="../conf", config_name="selfsup_experimental")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model = SelfSupervisedGANTraining(cfg)
    data_module = SelfSupervisedExperimentalDataModule(cfg)

    logger = MLFlowLogger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)
    callbacks = [
        RichModelSummary(max_depth=4),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ModelCheckpoint(
            save_top_k=2,
            save_last=True,
            monitor="Val/total",
            mode="min",
            auto_insert_metric_name=False,
        ),
        GradNormMonitor(),
    ]

    if cfg.profile:
        from lightning.pytorch.profilers import PyTorchProfiler

        Path("./profile").mkdir(parents=True, exist_ok=True)
        profiler = PyTorchProfiler(
            dirpath="./profile",
            export_to_chrome=True,
            profile_memory=True,
        )
        epochs = 3
        overfit_batches = 10
        torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        profiler = None
        epochs = cfg.epochs
        overfit_batches = cfg.overfit_batches

    device_count = torch.cuda.device_count()
    if cfg.strategy == "ddp" or (cfg.strategy == "auto" and device_count > 1):
        print(f"Using DDP strategy+NCCL with {device_count} GPUs")
        strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    else:
        strategy = "auto"

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        overfit_batches=overfit_batches,
        profiler=profiler,
        log_every_n_steps=cfg.get("log_every_n_steps", None),
    )

    trainer.fit(model=model, datamodule=data_module)

    if cfg.profile:
        torch.cuda.memory._dump_snapshot("./profile/memory_snapshot.json")

if __name__ == "__main__":
    main()
