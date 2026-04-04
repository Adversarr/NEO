"""
Car Pressure Simulation
h5dump -H data/shapenet_car/train.h5                                                                                                                              
HDF5 "data/shapenet_car/train.h5" {
GROUP "/" {
   ATTRIBUTE "grid_type" {
      DATATYPE  H5T_STRING {
         STRSIZE H5T_VARIABLE;
         STRPAD H5T_STR_NULLTERM;
         CSET H5T_CSET_UTF8;
         CTYPE H5T_C_S1;
      }
      DATASPACE  SCALAR
   }
   ATTRIBUTE "phys_dim" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SCALAR
   }
   DATASET "fx" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 789, 32186, 4 ) / ( 789, 32186, 4 ) }
   }
   DATASET "names" {
      DATATYPE  H5T_STRING {
         STRSIZE 39;
         STRPAD H5T_STR_NULLPAD;
         CSET H5T_CSET_ASCII;
         CTYPE H5T_C_S1;
      }
      DATASPACE  SIMPLE { ( 789 ) / ( 789 ) }
   }
   DATASET "pos" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 789, 32186, 3 ) / ( 789, 32186, 3 ) }
   }
   DATASET "surf" {
      DATATYPE  H5T_ENUM {
         H5T_STD_I8LE;
         "FALSE"            0;
         "TRUE"             1;
      }
      DATASPACE  SIMPLE { ( 789, 32186 ) / ( 789, 32186 ) }
   }
   DATASET "y" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 789, 32186, 4 ) / ( 789, 32186, 4 ) }
   }
}
}
"""

from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from g2pt.utils.mesh_feats import point_cloud_laplacian
from g2pt.data.transforms import normalize_pc
import h5py
import numpy as np
import torch

@dataclass
class ShapeNetCarSimulationDatasetConfig:
    data_dir: str = '/data/preprocessed_shapenetcar'
    train: str = 'train.hdf5'
    val: str = 'val.hdf5'
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2

class ShapeNetCarSimulationDataset(Dataset):
    _mean_feat = None
    _std_feat = None
    _mean_out = None
    _std_out = None

    def __init__(self, cfg: ShapeNetCarSimulationDatasetConfig, split: str = 'train'):
        self.cfg = cfg
        self.split = split
        self.data_dir = cfg.data_dir
        self.file = cfg.train if split == 'train' else cfg.val
        self.hf = h5py.File(f"{self.data_dir}/{self.file}", "r")
        if any(v is None for v in (ShapeNetCarSimulationDataset._mean_feat, ShapeNetCarSimulationDataset._std_feat, ShapeNetCarSimulationDataset._mean_out, ShapeNetCarSimulationDataset._std_out)):
            self._init_normalizer()

    def _init_normalizer(self):
        train_path = f"{self.data_dir}/{self.cfg.train}"
        with h5py.File(train_path, "r") as hf:
            x_ds = hf["fx"]
            y_ds = hf["y"]
            n_samples = x_ds.shape[0]
            sum_x = np.zeros((x_ds.shape[-1],), dtype=np.float64)
            sumsq_x = np.zeros_like(sum_x)
            cnt_x = 0
            sum_y = np.zeros((y_ds.shape[-1],), dtype=np.float64)
            sumsq_y = np.zeros_like(sum_y)
            cnt_y = 0
            for i in range(n_samples):
                x_i = np.array(x_ds[i], dtype=np.float64)
                y_i = np.array(y_ds[i], dtype=np.float64)
                sum_x += x_i.sum(axis=0)
                sumsq_x += (x_i * x_i).sum(axis=0)
                cnt_x += x_i.shape[0]
                sum_y += y_i.sum(axis=0)
                sumsq_y += (y_i * y_i).sum(axis=0)
                cnt_y += y_i.shape[0]
            mean_x = sum_x / max(cnt_x, 1)
            var_x = sumsq_x / max(cnt_x, 1) - mean_x * mean_x
            std_x = np.sqrt(np.maximum(var_x, 1e-12))
            mean_y = sum_y / max(cnt_y, 1)
            var_y = sumsq_y / max(cnt_y, 1) - mean_y * mean_y
            std_y = np.sqrt(np.maximum(var_y, 1e-12))
            ShapeNetCarSimulationDataset._mean_feat = mean_x.astype(np.float32)
            ShapeNetCarSimulationDataset._std_feat = std_x.astype(np.float32)
            ShapeNetCarSimulationDataset._mean_out = mean_y.astype(np.float32)
            ShapeNetCarSimulationDataset._std_out = std_y.astype(np.float32)
            print(f"mean_feat: {ShapeNetCarSimulationDataset._mean_feat}")
            print(f"std_feat: {ShapeNetCarSimulationDataset._std_feat}")
            print(f"mean_out: {ShapeNetCarSimulationDataset._mean_out}")
            print(f"std_out: {ShapeNetCarSimulationDataset._std_out}")

    def __len__(self):
        return int(self.hf["pos"].shape[0])

    def __getitem__(self, idx):
        pos = np.array(self.hf["pos"][idx], dtype=np.float32)
        surf = np.array(self.hf["surf"][idx], dtype=np.bool_)
        x = np.array(self.hf["fx"][idx], dtype=np.float32)
        y = np.array(self.hf["y"][idx], dtype=np.float32)

        # Normalize x
        x = (x - ShapeNetCarSimulationDataset._mean_feat) / ShapeNetCarSimulationDataset._std_feat

        pos_norm = normalize_pc(pos)
        full_pos = torch.tensor(pos_norm, dtype=torch.float32)
        surf_mask = torch.tensor(surf, dtype=torch.bool)
        surf_pos_np = normalize_pc(pos_norm[surf])
        _, M = point_cloud_laplacian(surf_pos_np.astype(np.float64))
        mass_np = M.diagonal().astype(np.float32).reshape(-1, 1)
        surf_pos = torch.tensor(surf_pos_np, dtype=torch.float32)
        surf_mass = torch.tensor(mass_np, dtype=torch.float32)
        full_feat = torch.tensor(x, dtype=torch.float32)
        out = torch.tensor(y, dtype=torch.float32)
        return {
            "surf": surf_mask,
            "surf_pos": surf_pos,
            "surf_mass": surf_mass,
            "full_pos": full_pos,
            "full_feat": full_feat,
            "out": out,
            "x_mean": ShapeNetCarSimulationDataset._mean_feat, # (4,)
            "x_std": ShapeNetCarSimulationDataset._std_feat,   # (4,)
            "y_mean": ShapeNetCarSimulationDataset._mean_out,  # (4,)
            "y_std": ShapeNetCarSimulationDataset._std_out,    # (4,)
        }

    def __del__(self):
        try:
            if hasattr(self, "hf") and self.hf:
                self.hf.close()
        except Exception:
            pass


class ShapeNetCarSimulationDataModule(LightningDataModule):
    def __init__(self, cfg: ShapeNetCarSimulationDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        self.train_set = ShapeNetCarSimulationDataset(self.cfg, split="train")
        self.val_set = ShapeNetCarSimulationDataset(self.cfg, split="val")

    def train_dataloader(self):
        kwargs = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
            "persistent_workers": True if self.cfg.num_workers > 0 else False,
        }
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return DataLoader(self.train_set, **kwargs)

    def val_dataloader(self):
        kwargs = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": False,
            "persistent_workers": True if self.cfg.num_workers > 0 else False,
        }
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return DataLoader(self.val_set, **kwargs)
