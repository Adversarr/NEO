from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from g2pt.data.transforms import normalize_pc


def _determine_segment(mesh_count: list[int], idx: int) -> tuple[int, int]:
    cum = 0
    for i, c in enumerate(mesh_count):
        if idx < cum + c:
            return i, idx - cum
        cum += c
    raise IndexError("Index out of bounds")


class CorrespondenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        enable_rotate: float = 0.0,
        mesh_glob: str | None = None,
        corr_glob: str | None = None,
        k_corr: int = 30,
    ) -> None:
        """Unified Dataset for Functional Map correspondence.

        Expects paired files in `data_dir` (created by preprocess_corr.py):
        - *_mesh.hdf5: verts(vlen float32), faces(vlen int32), vert_shapes, face_shapes
        - *_corr.hdf5: evals(vlen float32), evecs(vlen float32)+evec_shapes, hks(vlen float32)+hks_shapes, corres(vlen int32)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.enable_rotate = enable_rotate
        mesh_pat = mesh_glob or "*_mesh.hdf5"
        corr_pat = corr_glob or "*_corr.hdf5"
        self.mesh_files = sorted([str(p) for p in self.data_dir.glob(mesh_pat)])
        self.corr_files = sorted([str(p) for p in self.data_dir.glob(corr_pat)])
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory"
        assert len(self.mesh_files) > 0, "No *_mesh.hdf5 found"
        assert len(self.mesh_files) == len(self.corr_files), "Mesh and corr file counts mismatch"
        self.mesh_count: list[int] = []
        for m, c in zip(self.mesh_files, self.corr_files):
            with h5py.File(m, "r") as hf_m, h5py.File(c, "r") as hf_c:
                count = len(hf_m["verts"])  # number of meshes in this segment
                self.mesh_count.append(count)
                assert len(hf_c["evals"]) == count
                assert len(hf_c["evecs"]) == count
                assert len(hf_c["hks"]) == count
                assert len(hf_c["corres"]) == count
        self.opened_files: dict[str, h5py.File] = {}
        self.k_corr = k_corr

    def __len__(self) -> int:
        total_mesh = sum(self.mesh_count)
        return total_mesh * total_mesh

    def get_one(self, index: int) -> dict:
        seg_id, in_seg_idx = _determine_segment(self.mesh_count, index)
        mesh_f = self.mesh_files[seg_id]
        corr_f = self.corr_files[seg_id]
        if mesh_f not in self.opened_files:
            self.opened_files[mesh_f] = h5py.File(mesh_f, "r")
        if corr_f not in self.opened_files:
            self.opened_files[corr_f] = h5py.File(corr_f, "r")
        hf_m = self.opened_files[mesh_f]
        hf_c = self.opened_files[corr_f]

        v_flat = np.array(hf_m["verts"][in_seg_idx], dtype=np.float32)
        v_shape = tuple(hf_m["vert_shapes"][in_seg_idx].tolist())
        verts = v_flat.reshape(v_shape)
        mass = np.array(hf_c["mass"][in_seg_idx], dtype=np.float32)

        evals = np.array(hf_c["evals"][in_seg_idx], dtype=np.float32).clip(0., float('inf'))
        evec_flat = np.array(hf_c["evecs"][in_seg_idx], dtype=np.float32)
        e_shape = tuple(hf_c["evec_shapes"][in_seg_idx].tolist())
        evecs = evec_flat.reshape(e_shape)
        corres = np.array(hf_c["corres"][in_seg_idx], dtype=np.int32)

        points_tensor = torch.tensor(
            normalize_pc(verts, self.enable_rotate), dtype=torch.float32
        )
        return {
            "points": points_tensor,
            "mass": torch.tensor(mass, dtype=torch.float32),   # (npoints,)
            "evecs": torch.tensor(evecs, dtype=torch.float32), # (npoints, n_evecs)
            "evals": torch.tensor(evals, dtype=torch.float32), # (n_evecs,)
            "corres": torch.tensor(corres, dtype=torch.long),  # already 0-based
        }

    def __getitem__(self, index) -> dict:
        total_meshes = sum(self.mesh_count)
        first, second = index // total_meshes, index % total_meshes
        first_dict = self.get_one(first)
        second_dict = self.get_one(second)

        # Map to template
        c1, c2 = first_dict["corres"], second_dict["corres"]
        ev1a = first_dict["evecs"][c1, : self.k_corr]  # (npoints, k_corr)
        ev2a = second_dict["evecs"][c2, : self.k_corr]  # (npoints, k_corr)
        p1 = first_dict["points"][c1, :] # (npoints, 3)
        p2 = second_dict["points"][c2, :] # (npoints, 3)
        m1 = first_dict["mass"][c1] # (npoints,)
        m2 = second_dict["mass"][c2] # (npoints,)

        # Compute correspondence by lstsq
        C_gt = torch.linalg.lstsq(ev1a, ev2a).solution # (k_corr, k_corr)
        return {
            "p1": p1, "m1": m1.reshape(-1, 1),
            "p2": p2, "m2": m2.reshape(-1, 1),
            "C_gt": C_gt,
            # Required by correspondence computation:
            "evals1": first_dict["evals"][: self.k_corr],  # (k_corr,)
            "evals2": second_dict["evals"][: self.k_corr],  # (k_corr,)
            "evecs1": ev1a,  # (npoints, k_corr)
            "evecs2": ev2a,  # (npoints, k_corr)
        }


@dataclass
class UnifiedCorrDataModuleConfig:
    batch_size: int
    data_dir: str
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    limit_train: int = 0
    train_files: str = "*train_corr.hdf5"
    test_files: str = "*test_corr.hdf5"
    k_corr: int = 30


class UnifiedCorrDataModule(LightningDataModule):
    def __init__(self, cfg: UnifiedCorrDataModuleConfig):
        super().__init__()
        self.cfg = cfg

    def _make_dataset(self, split: str):
        corr_glob = self.cfg.train_files if split == "train" else self.cfg.test_files
        mesh_glob = corr_glob.replace("corr", "mesh")
        enable_rotate = self.cfg.enable_rotate if split == "train" else 0.0
        d = CorrespondenceDataset(
            data_dir=self.cfg.data_dir,
            enable_rotate=enable_rotate,
            corr_glob=corr_glob,
            mesh_glob=mesh_glob,
            k_corr=self.cfg.k_corr,
        )
        print(f"Len of {split} dataset = {len(d)}")
        return d

    def train_dataloader(self):
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = self._make_dataset("train")
        if self.cfg.limit_train and self.cfg.limit_train > 0:
            from torch.utils.data import Subset
            limit = min(self.cfg.limit_train, len(dataset))
            dataset = Subset(dataset, list(range(limit)))
        return DataLoader(
            dataset,
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
        dataset = self._make_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
        )

if __name__ == '__main__':
    dt = CorrespondenceDataset(
        data_dir='/data/processed_corr_FAUST',
        enable_rotate=0.0,
        corr_glob='*train_corr.hdf5',
        mesh_glob='*train_mesh.hdf5',
    )

    first = dt[0]
    for k, v in first.items():
        print(k, v.shape)
