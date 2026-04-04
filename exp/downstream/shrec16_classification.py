from pathlib import Path

import hydra
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger
import torch

from g2pt.data.datasets.unified_cls_datamod import UnifiedClsDataModule
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.classification import ClassificationTraining


@hydra.main(version_base=None, config_path="../conf", config_name="classification")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    datamod_cfg = cfg.datamod
    data_module = UnifiedClsDataModule(datamod_cfg)
    nc = getattr(data_module, "num_classes", datamod_cfg.n_class)
    print(f"⚙️ num_classes for classification: {nc}")
    model = ClassificationTraining(cfg, nc)

    logger = MLFlowLogger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)
    callbacks = [
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            save_top_k=2,
            save_last=True,
            monitor="Val/Loss",
            mode="min",
            auto_insert_metric_name=False,
        ),
        GradNormMonitor(),
    ]

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
        epochs = 3
        overfit_batches = 10
        torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        profiler = None
        epochs = cfg.epochs
        overfit_batches = cfg.overfit_batches

    device_count = torch.cuda.device_count()
    if cfg.strategy == "ddp" or (cfg.strategy == "auto" and device_count > 1):
        strategy = DDPStrategy(process_group_backend="nccl")
    else:
        strategy = "auto"

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        strategy=strategy,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        gradient_clip_val=cfg.gradient_clip_val,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        overfit_batches=overfit_batches,
        profiler=profiler,
        enable_model_summary=False,
    )

    trainer.fit(model=model, datamodule=data_module)

    if cfg.profile:
        torch.cuda.memory._dump_snapshot("memory.pickle")


if __name__ == "__main__":
    main()

