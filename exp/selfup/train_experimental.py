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
import torch

from g2pt.data.datasets.selfsup_experimental import SelfSupervisedExperimentalDataModule
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.selfsup_experimental import SelfSupervisedExperimentalTraining

@hydra.main(version_base=None, config_path="../conf", config_name="selfsup_experimental")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model = SelfSupervisedExperimentalTraining(cfg)
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

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        strategy=cfg.get("strategy", "auto"),
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        gradient_clip_val=cfg.gradient_clip_val,
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
