from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
from lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from g2pt.data.common import split
from g2pt.data.transforms import normalize_pc

def load_pc_mass_evec(sample: Path, point_idx: int, targ_dim: int, enable_rotate: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess point cloud, per-point mass, and eigenvectors for a sample.
    Returns `(pc_tensor, evec_tensor, mass_tensor)` ready for model consumption.
    """
    # 1. Load and center the point cloud.
    pc = np.load(sample / f"points_{point_idx}.npy")
    pc_tensor = torch.tensor(normalize_pc(pc, enable_rotate), dtype=torch.float32)

    # 2. Load and normalize the mass tensor.
    mass_np = np.load(sample / f"mass_{point_idx}.npy")
    mass_np = mass_np / (np.mean(mass_np) + 1e-6)
    mass_tensor = torch.tensor(mass_np, dtype=torch.float32).reshape(-1, 1)

    # 3. Load eigenvectors and apply mass-weighted normalization.
    evec_np = np.load(sample / f"evec_{point_idx}.npy")  # (N, k)
    evec_tensor = torch.tensor(evec_np, dtype=torch.float32)
    evec_tensor = evec_tensor[:, :targ_dim]
    evec_mass_norm = torch.sqrt(torch.sum(evec_tensor * evec_tensor * mass_tensor, dim=0, keepdim=True))
    evec_tensor = evec_tensor / (evec_mass_norm + 1e-6)
    return pc_tensor, evec_tensor, mass_tensor

class ProcessedEigvecDataset(Dataset):
    def __init__(
        self,
        samples: list[Path],
        per_mesh_count: int = 4,
        targ_dim: int = 16,
        enable_rotate: float = 0.0,
    ):
        super().__init__()
        self.samples = samples
        self.per_mesh_count = per_mesh_count
        self.targ_dim = targ_dim
        self.enable_rotate = enable_rotate
        if not self.samples:
            raise ValueError("No samples provided. Please provide a list of sample paths.")

    def __len__(self):
        return len(self.samples) * self.per_mesh_count

    def __getitem__(self, idx):
        mesh_idx = idx // self.per_mesh_count
        point_idx = idx % self.per_mesh_count
        sample = self.samples[mesh_idx]
        pc_tensor, evec_tensor, mass_tensor = load_pc_mass_evec(
            sample=sample,
            point_idx=point_idx,
            targ_dim=self.targ_dim,
            enable_rotate=self.enable_rotate,
        )
        return {"points": pc_tensor, "evecs": evec_tensor, "mass": mass_tensor}

@dataclass
class PretrainDataModuleConfig:
    data_dir: str
    batch_size: int
    targ_dim: int
    split_ratio: float = 0.9
    per_mesh_count: int = 4
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

class PretrainDataModule(LightningDataModule):
    def __init__(self, cfg: PretrainDataModuleConfig):
        super().__init__()
        self.data_dir = Path(cfg.data_dir)
        self.batch_size = cfg.batch_size
        self.targ_dim = cfg.targ_dim
        self.split_ratio = cfg.split_ratio
        self.per_mesh_count = cfg.per_mesh_count
        self.enable_rotate = cfg.enable_rotate
        self.workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.prefetch_factor = cfg.prefetch_factor

        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist.")
        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory {self.data_dir} is not a directory.")
        samples = list(self.data_dir.glob("*"))
        train_indices, val_indices = split(len(samples), self.split_ratio)
        self.train_samples = [samples[i] for i in train_indices]
        self.val_samples = [samples[i] for i in val_indices]
        print(f"Found {len(samples)} samples in {self.data_dir}")
        print(f"Training on {len(self.train_samples)} samples, validating on {len(self.val_samples)} samples.")

    def train_dataloader(self):
        mp_ctx = 'spawn' if self.workers > 0 else None
        return DataLoader(
            ProcessedEigvecDataset(
                self.train_samples,
                per_mesh_count=self.per_mesh_count,
                targ_dim=self.targ_dim,
                enable_rotate=self.enable_rotate,
            ),
            multiprocessing_context=mp_ctx,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,  # Keep workers alive for faster data loading.
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        mp_ctx = 'spawn' if self.workers > 0 else None
        return DataLoader(
            ProcessedEigvecDataset(
                self.val_samples,
                per_mesh_count=self.per_mesh_count,
                targ_dim=self.targ_dim,
                enable_rotate=self.enable_rotate,
            ),
            multiprocessing_context=mp_ctx,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )
