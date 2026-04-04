from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from g2pt.data.common import split
from g2pt.data.datasets.unified_h5 import UnifiedPreprocessedH5Dataset

@dataclass
class PreprocessedObjaverseLowpolyDataModuleConfig:
    """Configuration for PreprocessedObjaverseLowpolyDataModule"""

    batch_size: int
    data_dir: str
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    split_ratio: float = 0.9
    targ_dim: int = 96

class PreprocessedObjaverseLowpolyDataModule(LightningDataModule):
    """
    Lightning DataModule for preprocessed Objaverse Lowpoly dataset.
    Handles loading, splitting, and serving of preprocessed Objaverse Lowpoly data.
    """

    def __init__(self, cfg: PreprocessedObjaverseLowpolyDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        dataset = UnifiedPreprocessedH5Dataset(
            data_dir=cfg.data_dir,
            enable_rotate=cfg.enable_rotate,
            targ_dim=cfg.targ_dim,
        )

        per_mesh_count = dataset.per_mesh_count
        total = len(dataset)
        self.train_indices, self.val_indices = split(total // per_mesh_count, cfg.split_ratio, per_mesh_count)


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
