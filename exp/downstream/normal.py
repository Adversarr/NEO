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
import torch

from g2pt.data.datasets.unified_normal_h5 import UnifiedNormalH5DataModule
from g2pt.training.normal import NormalTraining


@hydra.main(version_base=None, config_path="../conf", config_name="normal")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    datamod_cfg = cfg.datamod
    data_module = UnifiedNormalH5DataModule(datamod_cfg)
    
    model = NormalTraining(cfg)

    from lightning.pytorch.loggers import MLFlowLogger
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
