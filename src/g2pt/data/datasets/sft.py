from dataclasses import dataclass
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data.dataset import Subset

from lightning import LightningDataModule
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from g2pt.data.common import split
from g2pt.data.transforms import normalize_pc
from g2pt.utils.gev import balance_stiffness, solve_gev_gt
from g2pt.utils.mesh_feats import load_and_process_mesh, point_cloud_laplacian, sample_points_uniformly
from g2pt.utils.rot import random_rotate_3d
from g2pt.utils.sparse import to_torch_sparse_csr
import h5py

from g2pt.data.datasets.selfsup_experimental import UnifiedPreprocessedH5MeshAdaptor_Exp, Geometry

@dataclass
class SupervisedFintuneDataModuleConfig:
    """Configuration for SupervisedFintuneDataModule"""

    name: str
    data_dir: str
    n_points: int
    batch_size: int
    enable_rotate: float
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    split_ratio: float
    target_k: int
    seed: int = 42


class SupervisedFinetuneDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    Computes Laplacian and eigenvectors on the fly.
    """

    def __init__(
        self,
        n_points: int,
        dataset_name: str,
        dataset_base_path: str,
        enable_rotate: float,
        target_k: int,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.dataset_name = dataset_name
        self.dataset_base_path = dataset_base_path
        self.adaptor = UnifiedPreprocessedH5MeshAdaptor_Exp(dataset_base_path, target_k)
        self.enable_rotate = enable_rotate
        self.target_k = target_k

    def __len__(self) -> int:
        return self.adaptor.len()

    def __getitem__(self, index):
        geometry = self.adaptor.get(index)
        points = sample_points_uniformly(geometry.verts, geometry.faces, self.n_points)
        L, M = point_cloud_laplacian(points)
        mass = M.diagonal()
        mass = mass / np.clip(mass.mean(), min=1e-3 / self.n_points)
        mass = mass.reshape(-1, 1)
        points_torch = torch.from_numpy(points)
        _, evecs = solve_gev_gt(points, k=self.target_k, L=L, M=M)
        evecs = torch.from_numpy(evecs)  # (n, k)
        evecs_norm = torch.sum(evecs**2 * torch.from_numpy(mass), dim=0, keepdim=True)  # (1, k)
        evecs = evecs / torch.sqrt(evecs_norm)  # (n, k)

        return {
            "points": torch.tensor(normalize_pc(points, self.enable_rotate), dtype=torch.float32),
            "evecs": evecs.to(dtype=torch.float32),
            "mass": torch.from_numpy(mass).to(dtype=torch.float32).reshape(-1, 1),
        }


class SupervisedFinetuneDataModule(LightningDataModule):
    """
    Lightning DataModule for supervised fine-tuning.
    Handles loading, splitting, and serving of SFT data.
    """

    def __init__(self, cfg: SupervisedFintuneDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        dataset = SupervisedFinetuneDataset(
            n_points=cfg.n_points,
            dataset_name=cfg.name,
            dataset_base_path=cfg.data_dir,
            enable_rotate=cfg.enable_rotate,
            target_k=cfg.target_k,
        )
        total = len(dataset)
        self.train_indices, self.val_indices = split(total, cfg.split_ratio, multiplier=1, seed=cfg.seed)

        print(f"Total samples: {total}")
        print(f"Train samples: {len(self.train_indices)}")
        print(f"Val samples: {len(self.val_indices)}")

    def train_dataloader(self):
        """Return the training dataloader"""
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = SupervisedFinetuneDataset(
            n_points=self.cfg.n_points,
            dataset_name=self.cfg.name,
            dataset_base_path=self.cfg.data_dir,
            enable_rotate=self.cfg.enable_rotate,
            target_k=self.cfg.target_k,
        )
        return DataLoader(
            Subset(dataset, self.train_indices),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            multiprocessing_context=mp_ctx,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
        )

    def val_dataloader(self):
        """Return the validation dataloader"""
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = SupervisedFinetuneDataset(
            n_points=self.cfg.n_points,
            dataset_name=self.cfg.name,
            dataset_base_path=self.cfg.data_dir,
            enable_rotate=0.0,
            target_k=self.cfg.target_k,
        )
        return DataLoader(
            Subset(dataset, self.val_indices),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
        )

