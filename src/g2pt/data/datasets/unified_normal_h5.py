from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from lightning import LightningDataModule
from g2pt.data.transforms import normalize_pc
from g2pt.data.common import split
from g2pt.utils.rot import random_rotate_3d

def rotate_with_normal(pc: np.ndarray, normal: np.ndarray, enable_rotate: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate point cloud with normal."""
    rot = random_rotate_3d(enable_rotate)
    pc = pc @ rot
    normal = normal @ rot
    return pc, normal

class UnifiedNormalH5Dataset(Dataset):
    """ShapeNet Normal dataset from HDF5 files.

    Expects files produced by exp/downstream/preprocess_normal.py that contain:
    - samples: (n_mesh, per_mesh_count, n_samples, 3)
    - normals: (n_mesh, per_mesh_count, n_samples, 3)
    - verts/faces/mass (optional)
    """
    def __init__(
        self,
        data_dir: str,
        enable_rotate: float = 0.0,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.enable_rotate = enable_rotate
        self.files = sorted([str(p) for p in self.data_dir.glob("*.hdf5")])
        assert len(self.files) > 0, f"No .hdf5 files found in {self.data_dir}"

        self.counts: list[int] = []
        found_per_mesh_counts = set()
        for f in self.files:
            with h5py.File(f, "r") as hf:
                # Shape is (n_mesh, per_mesh_count, n_samples, 3)
                n_mesh = len(hf["samples"])
                per_mesh_count = hf.attrs.get("per_mesh_count", 1)
                found_per_mesh_counts.add(per_mesh_count)
                self.counts.append(n_mesh * per_mesh_count)
        
        self.per_mesh_count = min(found_per_mesh_counts)
        self.opened: dict[str, h5py.File] = {}
        print(f"UnifiedNormalH5Dataset: {len(self.files)} files, {sum(self.counts)} samples")

    def __len__(self) -> int:
        return sum(self.counts)

    def __del__(self):
        """Close all opened HDF5 files"""
        self.close_files()
    
    def close_files(self):
        """Close all opened HDF5 files"""
        if hasattr(self, "opened"):
            for f in self.opened.values():
                try:
                    f.close()
                except Exception as e:
                    print(f"Error closing file {f}: {e}")
            self.opened.clear()

    def _determine_which_segment(self, idx: int) -> tuple[int, int]:
        """Determine which mesh file (segment) and index within that file"""
        cum_count = 0
        for batch_idx, count in enumerate(self.counts):
            if idx < cum_count + count:
                return batch_idx, idx - cum_count
            cum_count += count
        raise IndexError(f"Index {idx} out of bounds for dataset")
    
    def _ensure_file(self, file_path: str) -> h5py.File:
        if file_path not in self.opened:
            self.opened[file_path] = h5py.File(file_path, "r")
        return self.opened[file_path]

    def __getitem__(self, index: int) -> dict:
        seg_id, in_idx = self._determine_which_segment(index)
        hf = self._ensure_file(self.files[seg_id])

        per_mesh_count = hf.attrs.get("per_mesh_count", 1)
        mesh_idx = in_idx // per_mesh_count
        sub_idx = in_idx % per_mesh_count

        # Load samples and normals
        # hf["samples"] shape: (n_mesh, per_mesh_count, n_points, 3)
        pts = np.array(hf["samples"][mesh_idx, sub_idx], dtype=np.float32)
        nrms = np.array(hf["normals"][mesh_idx, sub_idx], dtype=np.float32)

        # Normalize point cloud (centering/scaling/rotation)
        pts = normalize_pc(pts, 0)
        
        # We need to compute mass if it's not precomputed. 
        # For simplicity, we can assume uniform mass if not present.
        if "mass" in hf:
            mass = np.array(hf["mass"][mesh_idx, sub_idx], dtype=np.float32)
        else:
            mass = np.ones((pts.shape[0],), dtype=np.float32)
        
        mass = mass.reshape(-1, 1)

        return {
            "points": torch.tensor(pts, dtype=torch.float32),
            "normals": torch.tensor(nrms, dtype=torch.float32),
            "mass": torch.tensor(mass, dtype=torch.float32),
        }


@dataclass
class UnifiedNormalH5DataModuleConfig:
    batch_size: int
    data_dir: str
    name: str = "normal"
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    limit_train: int = 0
    split_ratio: float = 0.9


class UnifiedNormalH5DataModule(LightningDataModule):
    def __init__(self, cfg: UnifiedNormalH5DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        dataset = UnifiedNormalH5Dataset(
            data_dir=cfg.data_dir,
            enable_rotate=cfg.enable_rotate,
        )
        per_mesh_count = dataset.per_mesh_count
        total = len(dataset)
        self.train_indices, self.val_indices = split(total // per_mesh_count, cfg.split_ratio, per_mesh_count)

        print(f"Total samples: {total}")
        print(f"Train samples: {len(self.train_indices)}")
        print(f"Val samples: {len(self.val_indices)}")

    def train_dataloader(self):
        dataset = UnifiedNormalH5Dataset(
            data_dir=self.cfg.data_dir,
            enable_rotate=self.cfg.enable_rotate,
        )
        indices = self.train_indices
        if self.cfg.limit_train > 0:
            limit = min(self.cfg.limit_train, len(indices))
            indices = indices[:limit]
        
        return DataLoader(
            Subset(dataset, indices),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )

    def val_dataloader(self):
        dataset = UnifiedNormalH5Dataset(
            data_dir=self.cfg.data_dir,
            enable_rotate=0.0,
        )
        return DataLoader(
            Subset(dataset, self.val_indices),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
