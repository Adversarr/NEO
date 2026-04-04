from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from g2pt.data.common import split
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.rot import random_rotate_3d
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly

import h5py
import scipy.sparse.linalg as la
import warnings


class ObjaverseLowpolyDataset(Dataset):
    def __init__(self, data_dir: str, enable_rotate: float, n_sample_points: int, targ_dim: int):
        self.data_dir = Path(data_dir)
        self.enable_rotate = enable_rotate
        self.n_sample_points = n_sample_points
        self.targ_dim = targ_dim

        vert_files = sorted(self.data_dir.glob("vert_*.hdf5"))
        face_files = sorted(self.data_dir.glob("face_*.hdf5"))
        vmap = {p.stem.split("_")[1]: p for p in vert_files}
        fmap = {p.stem.split("_")[1]: p for p in face_files}

        self.items: List[Tuple[np.ndarray, np.ndarray]] = []
        for key in sorted(set(vmap.keys()) & set(fmap.keys())):
            vp = vmap[key]
            fp = fmap[key]
            with h5py.File(vp, "r") as hv, h5py.File(fp, "r") as hf:
                v_shapes = np.array(hv["vert_shapes"], dtype=np.int32)
                f_shapes = np.array(hf["face_shapes"], dtype=np.int32)
                verts_ds = hv["verts"]
                faces_ds = hf["faces"]
                count = min(len(verts_ds), len(faces_ds))
                for i in range(count):
                    v_shape = tuple(v_shapes[i].tolist())
                    f_shape = tuple(f_shapes[i].tolist())
                    v_flat = np.array(verts_ds[i], dtype=np.float32)
                    f_flat = np.array(faces_ds[i], dtype=np.int32)
                    verts = v_flat.reshape(v_shape)
                    faces = f_flat.reshape(f_shape)
                    self.items.append((verts, faces))

        if not self.items:
            raise ValueError(f"No HDF5 vert/face pairs found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        verts, faces = self.items[idx]

        points = sample_points_uniformly(verts, faces, self.n_sample_points)
        points = points - np.mean(points, axis=0, keepdims=True)
        if self.enable_rotate > 0:
            rot = random_rotate_3d(self.enable_rotate)
            points = points @ rot.T.astype(points.dtype)
        max_abs = np.max(np.abs(points))
        points = points / (max_abs + 1e-12)

        # TODO: Add evec and mass.
        L, M = point_cloud_laplacian(points)
        L, M = balance_stiffness(L, M, delta=1, k=self.targ_dim)
        _, evecs = la.eigsh(L, k=self.targ_dim, M=M, which="SM")
        mass = M.diagonal()
        return {
            "points": torch.tensor(points, dtype=torch.float32),
            "evecs": torch.tensor(evecs[:, : self.targ_dim], dtype=torch.float32),
            "mass": torch.tensor(mass.reshape(-1, 1), dtype=torch.float32),
        }


@dataclass
class ObjaverseLowpolyDataModuleConfig:
    data_dir: str
    batch_size: int
    n_sample_points: int
    enable_rotate: float = 0.0
    split_ratio: float = 0.9
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    targ_dim: int = 3


class ObjaverseLowpolyDataModule(LightningDataModule):
    def __init__(self, cfg: ObjaverseLowpolyDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = ObjaverseLowpolyDataset(
            data_dir=cfg.data_dir,
            enable_rotate=cfg.enable_rotate,
            n_sample_points=cfg.n_sample_points,
            targ_dim=cfg.targ_dim,
        )
        total = len(self.dataset)
        self.train_indices, self.val_indices = split(total, cfg.split_ratio)

        if self.cfg.targ_dim > 512:
            warnings.warn(
                f"targ_dim={self.cfg.targ_dim} > 512, which may cause performance issues. Consider reducing targ_dim."
            )

    def train_dataloader(self):
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        return DataLoader(
            Subset(self.dataset, self.train_indices),
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
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        return DataLoader(
            Subset(self.dataset, self.val_indices),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
            prefetch_factor=self.cfg.prefetch_factor,
        )
