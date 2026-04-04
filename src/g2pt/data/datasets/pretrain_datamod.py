from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from g2pt.data.common import split
from g2pt.data.datasets.unified_h5 import UnifiedPreprocessedH5Dataset

@dataclass
class PretrainingDataModuleConfig:
    """Configuration for PretrainingDataModule"""

    batch_size: int
    data_dir: str
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    split_ratio: float = 0.9
    targ_dim: int = 96
    recursive: bool = False

class PretrainDataModule(LightningDataModule):
    """
    Lightning DataModule for pretraining on unified preprocessed HDF5 dataset.
    Handles loading, splitting, and serving of unified preprocessed HDF5 data.
    """

    def __init__(self, cfg: PretrainingDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        dataset = UnifiedPreprocessedH5Dataset(
            data_dir=cfg.data_dir,
            enable_rotate=cfg.enable_rotate,
            targ_dim=cfg.targ_dim,
            recursive=cfg.recursive,
        )

        per_mesh_count = dataset.per_mesh_count
        total = len(dataset)
        self.train_indices, self.val_indices = split(total // per_mesh_count, cfg.split_ratio, per_mesh_count)

        if cfg.recursive:
            print(f"📂 Recursively found {len(dataset.mesh_files)} files")
        else:
            print(f"🔍 Found {len(dataset.mesh_files)} files")

        self.recursive = cfg.recursive
        print(f"Total samples: {total}")
        print(f"Train samples: {len(self.train_indices)}")
        print(f"Val samples: {len(self.val_indices)}")

    def train_dataloader(self):
        """Return the training dataloader"""
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = UnifiedPreprocessedH5Dataset(
            data_dir=self.cfg.data_dir,
            enable_rotate=self.cfg.enable_rotate,
            targ_dim=self.cfg.targ_dim,
            recursive=self.recursive,
        )
        return DataLoader(
            Subset(dataset, self.train_indices),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            multiprocessing_context=mp_ctx,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Return the validation dataloader"""
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = UnifiedPreprocessedH5Dataset(
            data_dir=self.cfg.data_dir,
            enable_rotate=0,
            targ_dim=self.cfg.targ_dim,
            recursive=self.recursive,
        )
        return DataLoader(
            Subset(dataset, self.val_indices),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            pin_memory=False,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
