import json
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

from g2pt.data.datasets.partnet_pc_datamod import PartNetPCDataModule
from g2pt.data.datasets.unified_seg_datamod import UnifiedSegDataModule
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.segment import SegmentTraining


@hydra.main(version_base=None, config_path="../conf", config_name="segment")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    datamod_cfg = cfg.datamod
    dm_name = cfg.datamod.name
    if dm_name == 'partnet':
        data_module = PartNetPCDataModule(datamod_cfg)
    else:
        data_module = UnifiedSegDataModule(datamod_cfg)
    nc = data_module.num_classes
    print(f'⚙️ num_classes for segmentation: {nc}')
    model = SegmentTraining(cfg, nc)

    logger = MLFlowLogger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)
    callbacks = [
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="Val/Loss",
            mode="min",
            auto_insert_metric_name=False,
        ),
    ]
    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        strategy=cfg.strategy,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        gradient_clip_val=cfg.gradient_clip_val,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model=model, datamodule=data_module)
    output = trainer.validate(model=model, datamodule=data_module)
    print(output)

    if cfg.export_json:
        path = str(cfg.export_json)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()