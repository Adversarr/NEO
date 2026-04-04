from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule
from torch.nn import ModuleList
from torch.utils.data import DataLoader, Dataset, Subset

from g2pt.data.common import split
from g2pt.data.transforms import normalize_pc
from g2pt.metrics import get_metric
from g2pt.neuralop.model import get_model
from g2pt.training.common import create_optimizer_and_scheduler
from g2pt.utils.ortho_operations import qr_orthogonalization


class PretrainTetH5Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        enable_rotate: float = 0.0,
        targ_dim: int = 96,
        recursive: bool = False,
        return_mesh: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.enable_rotate = float(enable_rotate)
        self.targ_dim = int(targ_dim)
        self.return_mesh = bool(return_mesh)
        self.recursive = bool(recursive)

        if recursive:
            self.mesh_files = []
            self.samples_evecs_files = []
            for subfolder in self.data_dir.glob("*"):
                self.mesh_files.extend([str(path) for path in subfolder.glob("*_mesh.hdf5")])
                self.samples_evecs_files.extend([str(path) for path in subfolder.glob("*_samples_evecs.hdf5")])
            self.mesh_files.sort()
            self.samples_evecs_files.sort()
        else:
            self.mesh_files = sorted([str(path) for path in self.data_dir.glob("*_mesh.hdf5")])
            self.samples_evecs_files = sorted([str(path) for path in self.data_dir.glob("*_samples_evecs.hdf5")])

        if len(self.mesh_files) == 0:
            raise ValueError(f"No *_mesh.hdf5 files found in {self.data_dir}")
        if len(self.samples_evecs_files) == 0:
            raise ValueError(f"No *_samples_evecs.hdf5 files found in {self.data_dir}")
        if len(self.mesh_files) != len(self.samples_evecs_files):
            raise ValueError("Number of mesh and samples_evecs files must match")

        found_per_mesh_counts: set[int] = set()
        found_k: set[int] = set()
        self.mesh_count: list[int] = []
        for mesh_file in self.mesh_files:
            with h5py.File(mesh_file, "r") as hf:
                self.mesh_count.append(int(len(hf["verts"])))

        for samples_file in self.samples_evecs_files:
            with h5py.File(samples_file, "r") as hf:
                found_per_mesh_counts.add(int(hf.attrs["per_mesh_count"]))
                found_k.add(int(hf.attrs["k"]))

        self.per_mesh_count = min(found_per_mesh_counts) if found_per_mesh_counts else 1
        self.k = min(found_k) if found_k else self.targ_dim
        if self.k < self.targ_dim:
            raise ValueError(f"k={self.k} < targ_dim={self.targ_dim}")

        self.opened_files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return sum(self.mesh_count) * self.per_mesh_count

    def __del__(self) -> None:
        self.close_files()

    def close_files(self) -> None:
        for f in self.opened_files.values():
            try:
                f.close()
            except Exception:
                pass
        self.opened_files.clear()

    def _ensure_file(self, file_path: str) -> h5py.File:
        if file_path not in self.opened_files:
            self.opened_files[file_path] = h5py.File(file_path, "r")
        return self.opened_files[file_path]

    def _determine_which_segment(self, idx: int) -> tuple[int, int]:
        cum_count = 0
        for batch_idx, count in enumerate(self.mesh_count):
            if idx < cum_count + count:
                return batch_idx, idx - cum_count
            cum_count += count
        raise IndexError(f"Index {idx} out of bounds for dataset")

    def __getitem__(self, idx: int) -> dict:
        mesh_idx = idx // self.per_mesh_count
        subsample_idx = idx % self.per_mesh_count
        seg_id, in_seg_idx = self._determine_which_segment(mesh_idx)
        samples_evec_file = self._ensure_file(self.samples_evecs_files[seg_id])

        points = np.array(samples_evec_file["samples"][in_seg_idx, subsample_idx, :, :], dtype=np.float32)
        evecs = np.array(
            samples_evec_file["evecs"][in_seg_idx, subsample_idx, :, : self.targ_dim],
            dtype=np.float32,
        )
        mass = np.array(samples_evec_file["mass"][in_seg_idx, subsample_idx, :], dtype=np.float32)

        result = {
            "points": torch.tensor(normalize_pc(points, self.enable_rotate), dtype=torch.float32),
            "evecs": torch.from_numpy(evecs).to(dtype=torch.float32),
            "mass": torch.from_numpy(mass).to(dtype=torch.float32).reshape(-1, 1),
        }

        if self.return_mesh:
            mesh_file = self._ensure_file(self.mesh_files[seg_id])
            v_shape = tuple(mesh_file["vert_shapes"][in_seg_idx].tolist())
            f_shape = tuple(mesh_file["face_shapes"][in_seg_idx].tolist())
            v_flat = np.array(mesh_file["verts"][in_seg_idx], dtype=np.float32)
            f_flat = np.array(mesh_file["faces"][in_seg_idx], dtype=np.int32)
            result["verts"] = torch.from_numpy(v_flat.reshape(v_shape)).to(dtype=torch.float32)
            result["faces"] = torch.from_numpy(f_flat.reshape(f_shape)).to(dtype=torch.int32)

        return result


class PretrainTetDataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.data_dir = getattr(cfg, "data_dir", None)
        if self.data_dir is None:
            raise ValueError("datamod.data_dir is required")
        self.batch_size = int(getattr(cfg, "batch_size", 1))
        self.enable_rotate = float(getattr(cfg, "enable_rotate", 0.0))
        self.num_workers = int(getattr(cfg, "num_workers", 4))
        self.pin_memory = bool(getattr(cfg, "pin_memory", True))
        self.prefetch_factor = int(getattr(cfg, "prefetch_factor", 4))
        self.split_ratio = float(getattr(cfg, "split_ratio", 0.9))
        self.targ_dim = int(getattr(cfg, "targ_dim", 96))
        self.recursive = bool(getattr(cfg, "recursive", False))

        dataset = PretrainTetH5Dataset(
            data_dir=self.data_dir,
            enable_rotate=self.enable_rotate,
            targ_dim=self.targ_dim,
            recursive=self.recursive,
        )
        per_mesh_count = dataset.per_mesh_count
        total = len(dataset)
        self.train_indices, self.val_indices = split(total // per_mesh_count, self.split_ratio, per_mesh_count)

        print(f"Total samples: {total}")
        print(f"Train samples: {len(self.train_indices)}")
        print(f"Val samples: {len(self.val_indices)}")

    def train_dataloader(self):
        mp_ctx = "spawn" if self.num_workers > 0 else None
        dataset = PretrainTetH5Dataset(
            data_dir=self.data_dir,
            enable_rotate=self.enable_rotate,
            targ_dim=self.targ_dim,
            recursive=self.recursive,
        )
        return DataLoader(
            Subset(dataset, self.train_indices),
            batch_size=self.batch_size,
            shuffle=True,
            multiprocessing_context=mp_ctx,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        mp_ctx = "spawn" if self.num_workers > 0 else None
        dataset = PretrainTetH5Dataset(
            data_dir=self.data_dir,
            enable_rotate=0.0,
            targ_dim=self.targ_dim,
            recursive=self.recursive,
        )
        return DataLoader(
            Subset(dataset, self.val_indices),
            batch_size=self.batch_size,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )


class PretrainTetTraining(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.targ_dim: int = int(cfg.get("targ_dim", 16))
        self.targ_dim_model: int = int(cfg.get("targ_dim_model", self.targ_dim))
        self.model = get_model(3, 3, self.targ_dim_model, cfg.model)
        self.model.reset_parameters()

        self.optimizer_config = cfg.optimizer
        self.scheduler_config = cfg.scheduler

        self.val_losses = ModuleList()
        self.val_weights: list[float] = []
        for val_loss_args in cfg.validation_metrics:
            kwargs = val_loss_args.get("kwargs", {})
            self.val_losses.append(get_metric(name=val_loss_args["name"], **kwargs))
            self.val_weights.append(float(val_loss_args.get("weight", 1.0)))

        self.losses = ModuleList()
        self.weights: list[float] = []
        for metric_args in cfg.training_metrics:
            kwargs = metric_args.get("kwargs", {})
            self.losses.append(get_metric(name=metric_args["name"], **kwargs))
            self.weights.append(float(metric_args.get("weight", 1.0)))

        if cfg.get("compile", False):
            self.model = torch.compile(self.model)

    def forward(self, x: torch.Tensor, mass: torch.Tensor):
        y = self.model(x, x, mass)
        with torch.autocast(device_type="cuda", enabled=False):
            return qr_orthogonalization(y, mass), y

    def _extract_batch(self, batch):
        x = batch["points"]
        mass = batch["mass"]
        y = batch["evecs"]
        return x, y, mass

    def training_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original = self(x, mass)
        total: torch.Tensor = 0  # type: ignore

        with torch.autocast(device_type=x.device.type, enabled=False):
            y_hat = y_hat.to(torch.float32)
            for loss_fn, weight in zip(self.losses, self.weights):
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                if weight > 0:
                    total = total + loss_value * weight
                self.log(f"Train/{type(loss_fn).__name__}", loss_value, prog_bar=True)
            self.log("Train/total", total, prog_bar=False)

        self.log("Train/mesh_size", x.shape[1], prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        x, y, mass = self._extract_batch(batch)
        y_hat, y_original = self(x, mass)
        with torch.no_grad():
            total: torch.Tensor = 0  # type: ignore
            for loss_fn, weight in zip(self.val_losses, self.val_weights):
                loss_value = loss_fn(pred=y_hat, target=y, mass=mass, y_original=y_original)
                total = total + loss_value * weight
                self.log(f"Val/{type(loss_fn).__name__}", loss_value, sync_dist=True)
            self.log("Val/total", total, prog_bar=True, on_step=False, sync_dist=True)
        return total

    def configure_optimizers(self):  # type: ignore
        optimizer, scheduler = create_optimizer_and_scheduler(
            module=self.model,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
