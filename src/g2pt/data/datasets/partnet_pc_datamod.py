from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from g2pt.data.transforms import normalize_pc


class PartNetPointCloudDataset(Dataset):
    """PartNet point-cloud dataset with precomputed lumped mass.

    Expects files produced by exp/downstream/preprocess_partnet.py that contain
    datasets: points (n_mesh, n_points, 3), labels (n_mesh, n_points), mass (n_mesh, n_points).
    Returns dict with keys: points(float32), mass(float32, shape (n_points,1)), labels(int64).
    """
    def __init__(
        self,
        data_dir: str,
        n_points: int = 1024,
        enable_rotate: float = 0.0,
        split_glob: str | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.enable_rotate = enable_rotate
        pat = split_glob or "*_pc.hdf5"
        self.files = sorted([str(p) for p in self.data_dir.glob(pat)])
        assert len(self.files) > 0, f"No files matched {pat} in {self.data_dir}"

        self.counts: list[int] = []
        known_n_classes: int = 0
        max_label_seen: int = -1
        for f in self.files:
            with h5py.File(f, "r") as hf:
                self.counts.append(len(hf["points"]))
                if 'num_classes' in hf.attrs:
                    known_n_classes = int(hf.attrs['num_classes'])
                # track max label to infer classes when attribute missing
                if "labels" in hf.keys():
                    lbls = np.array(hf["labels"], dtype=np.int64)
                    if lbls.size > 0:
                        max_label_seen = max(max_label_seen, int(lbls.max()))
        if known_n_classes > 0:
            self.num_classes = known_n_classes
        else:
            self.num_classes = int(max_label_seen + 1) if max_label_seen >= 0 else 0
        self.opened: dict[str, h5py.File] = {}
        # Precompute class weights across entire dataset
        if self.num_classes and self.num_classes > 0:
            counts = np.zeros(self.num_classes, dtype=np.int64)
            for f in self.files:
                with h5py.File(f, "r") as hf:
                    seg_count = len(hf["labels"]) if "labels" in hf.keys() else 0
                    for i in range(seg_count):
                        lbl = np.array(hf["labels"][i], dtype=np.int64).reshape(-1)
                        for c in range(self.num_classes):
                            counts[c] += int((lbl == c).sum())
            eps = 1e-6
            inv_freq = 1.0 / (counts.astype(np.float64) + eps)
            inv_freq *= (self.num_classes / inv_freq.sum()) if inv_freq.sum() > 0 else 1.0
            self.class_weights = inv_freq.astype(np.float32)
        else:
            self.class_weights = np.array([], dtype=np.float32)
        print(f"PartNetPointCloudDataset: {self.num_classes} classes, {len(self.files)} files, {sum(self.counts)} samples")

    def __len__(self) -> int:
        return sum(self.counts)

    def __getitem__(self, index: int) -> dict:
        """Return one sample matching unified_seg_datamod.py outputs (points/mass/labels)."""
        cum = 0
        seg_id = 0
        for i, c in enumerate(self.counts):
            if index < cum + c:
                seg_id = i
                in_idx = index - cum
                break
            cum += c
        f = self.files[seg_id]
        if f not in self.opened:
            self.opened[f] = h5py.File(f, "r")
        hf = self.opened[f]

        pts = np.array(hf["points"][in_idx], dtype=np.float32)
        lbl = np.array(hf["labels"][in_idx], dtype=np.int64)
        mass = np.array(hf["mass"][in_idx], dtype=np.float32) # relative weight for each point

        pts = normalize_pc(pts, self.enable_rotate)
        mass = mass.reshape(-1, 1)

        if self.n_points > 0 and pts.shape[0] > self.n_points:
            choice = np.random.choice(pts.shape[0], self.n_points, replace=False)
            pts = pts[choice]
            lbl = lbl[choice]
            mass = mass[choice]

        return {
            "points": torch.tensor(pts, dtype=torch.float32),
            "labels": torch.tensor(lbl, dtype=torch.long),
            "mass": torch.tensor(mass, dtype=torch.float32),
            "class_weights": torch.tensor(self.class_weights, dtype=torch.float32),
        }


@dataclass
class PartNetPCDataModuleConfig:
    batch_size: int
    data_dir: str
    n_points: int = 1024
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    limit_train: int = 0
    train_glob: str = "train-*_pc.hdf5"
    test_glob: str = "val-*_pc.hdf5"


class PartNetPCDataModule(LightningDataModule):
    """Lightning DataModule for PartNet point-cloud segmentation.

    Provides train/val loaders yielding batches compatible with g2pt/training/segment.py.
    """
    def __init__(self, cfg: PartNetPCDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes: int = self._infer_num_classes()

    def _infer_num_classes(self) -> int:
        base = Path(self.cfg.data_dir)
        # prefer train; fallback to test
        candidates = list(sorted(base.glob(self.cfg.train_glob)))
        if not candidates:
            candidates = list(sorted(base.glob(self.cfg.test_glob)))
        if not candidates:
            raise RuntimeError(f"No classification label files found under {base} with patterns {self.cfg.train_glob} / {self.cfg.test_glob}")
        first = candidates[0]
        with h5py.File(str(first), "r") as hf:
            if "num_classes" in hf.attrs:
                return int(hf.attrs["num_classes"])  # preferred
            if "n_class" in hf.attrs:
                return int(hf.attrs["n_class"])  # legacy
            cls = np.array(hf["cls_labels"], dtype=np.int32)
            if cls.size == 0:
                raise RuntimeError(f"Empty cls_labels in {first}; cannot infer num_classes")
            return int(cls.max() + 1)

    def _make_dataset(self, split: str):
        """Construct dataset for the given split."""
        enable_rotate = self.cfg.enable_rotate if split == "train" else 0.0
        glob = self.cfg.train_glob if split == "train" else self.cfg.test_glob
        n_points = self.cfg.n_points if split == "train" else 0
        return PartNetPointCloudDataset(
            data_dir=self.cfg.data_dir,
            n_points=n_points,
            enable_rotate=enable_rotate,
            split_glob=glob,
        )

    def train_dataloader(self):
        """Create the training DataLoader."""
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
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )

    def val_dataloader(self):
        """Create the validation DataLoader."""
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = self._make_dataset("test")
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
