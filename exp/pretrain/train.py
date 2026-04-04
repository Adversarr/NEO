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
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger
import torch

from g2pt.data.datasets.merge import PreprocessedMergeDataModule
from g2pt.data.datasets.shapenet_h5 import PreprocessedShapeNetDataModule
from g2pt.training.callbacks.bottleneck_analyzer import BottleneckAnalyzer
from g2pt.training.pretrain import PretrainTraining
from g2pt.data.datasets.shapenet import PretrainDataModule
from g2pt.data.datasets.objaverse_lowpoly import ObjaverseLowpolyDataModule, ObjaverseLowpolyDataModuleConfig
from g2pt.data.datasets.objaverse_lowpoly_h5 import PreprocessedObjaverseLowpolyDataModule
from g2pt.data.datasets.sft import SupervisedFinetuneDataModule, SupervisedFintuneDataModuleConfig
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.common import load_partial_state_dict_strict


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg):
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model = PretrainTraining(cfg)
    datamod_cfg = cfg.datamod
    name = getattr(datamod_cfg, "name", None)
    if name == "processed_shapenet":
        data_module = PretrainDataModule(datamod_cfg)
    elif name == "processed_objaverse_lowpoly":
        data_module = ObjaverseLowpolyDataModule(
            ObjaverseLowpolyDataModuleConfig(
                data_dir=datamod_cfg.data_dir,
                batch_size=datamod_cfg.batch_size,
                n_sample_points=datamod_cfg.n_sample_points,
                enable_rotate=getattr(datamod_cfg, "enable_rotate", 0.0),
                split_ratio=getattr(datamod_cfg, "split_ratio", 0.9),
                num_workers=getattr(datamod_cfg, "num_workers", 4),
                pin_memory=getattr(datamod_cfg, "pin_memory", True),
                prefetch_factor=getattr(datamod_cfg, "prefetch_factor", 4),
                targ_dim=getattr(datamod_cfg, "targ_dim", cfg.get("targ_dim", 16)),
            )
        )
    elif name == 'mixed':
        data_module = PreprocessedMergeDataModule(datamod_cfg)
    elif name == "shapenet_h5":
        data_module = PreprocessedShapeNetDataModule(datamod_cfg)
    elif name == "objaverse_lowpoly_h5":
        data_module = PreprocessedObjaverseLowpolyDataModule(datamod_cfg)
    elif name == "sft":
        data_module = SupervisedFinetuneDataModule(
            SupervisedFintuneDataModuleConfig(
                name=datamod_cfg.name,
                data_dir=datamod_cfg.data_dir,
                n_points=datamod_cfg.n_points,
                batch_size=datamod_cfg.batch_size,
                enable_rotate=getattr(datamod_cfg, "enable_rotate", 0.0),
                num_workers=getattr(datamod_cfg, "num_workers", 4),
                pin_memory=getattr(datamod_cfg, "pin_memory", True),
                prefetch_factor=getattr(datamod_cfg, "prefetch_factor", 2),
                split_ratio=getattr(datamod_cfg, "split_ratio", 0.9),
                target_k=getattr(datamod_cfg, "target_k", 16),
            )
        )

    else:
        raise ValueError("Unknown datamod config: please set 'name' or required keys.")

    if cfg.get("ckpt_pretrain", None) is not None:
        print(f"🎉 Load pretrained model params (strict partial) from {cfg.get('ckpt_pretrain')}")
        load_partial_state_dict_strict(model.model, ckpt_path=cfg.get("ckpt_pretrain"), key_prefix='model.', strict=False)

    logger = MLFlowLogger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)
    callbacks = [
        RichModelSummary(max_depth=4),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            save_top_k=2,
            save_last=True,
            monitor="Val/total",
            mode="min",
            auto_insert_metric_name=False,
        ),
        GradNormMonitor(),
        # BottleneckAnalyzer(),
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
        print(f"Using DDP strategy+NCCL with {device_count} GPUs")
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
    )

    ckpt_path = cfg.get("prev_ckpt_path", None)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    if cfg.profile:
        torch.cuda.memory._dump_snapshot("memory.pickle")


if __name__ == "__main__":
    main()
