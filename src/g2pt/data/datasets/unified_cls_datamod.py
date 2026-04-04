from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from g2pt.utils.mesh_feats import point_cloud_laplacian, mesh_laplacian, sample_points_uniformly
from g2pt.data.transforms import normalize_pc


def _determine_segment(mesh_count: list[int], idx: int) -> tuple[int, int]:
    cum = 0
    for i, c in enumerate(mesh_count):
        if idx < cum + c:
            return i, idx - cum
        cum += c
    raise IndexError("Index out of bounds")


class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        n_class: int,
        n_points: int = 1024,
        enable_rotate: float = 0.0,
        seed: int | None = None,
        labels_glob: str | None = None,
        mesh_glob: str | None = None,
    ) -> None:
        """Unified Dataset for Classification task (point cloud sampling).

        Expects paired files in `data_dir`:
        - *_mesh.hdf5: verts(vlen float32), faces(vlen int32), vert_shapes, face_shapes
        - *_cls.hdf5:  cls_labels (num_meshes,) int32 class indices
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.n_class = n_class
        self.n_points = n_points
        self.enable_rotate = enable_rotate
        self.seed = seed
        mesh_pat = mesh_glob or "*_mesh.hdf5"
        labels_pat = labels_glob or "*_cls.hdf5"
        self.mesh_files = sorted([str(p) for p in self.data_dir.glob(mesh_pat)])
        self.labels_files = sorted([str(p) for p in self.data_dir.glob(labels_pat)])
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory"
        assert len(self.mesh_files) > 0, "No *_mesh.hdf5 found"
        assert len(self.mesh_files) == len(self.labels_files), "Mesh and label file counts mismatch"
        self.mesh_count: list[int] = []
        for m, l in zip(self.mesh_files, self.labels_files):
            with h5py.File(m, "r") as hf_m, h5py.File(l, "r") as hf_l:
                count = len(hf_m["verts"])  # number of meshes in this segment
                self.mesh_count.append(count)
                assert len(hf_l["cls_labels"]) == count, f"cls_labels length mismatch in {l}"
        self.opened_files: dict[str, h5py.File] = {}
        counts = np.zeros(self.n_class, dtype=np.int64)
        for lpath in self.labels_files:
            with h5py.File(lpath, "r") as hf_l:
                cls_labels = np.array(hf_l["cls_labels"], dtype=np.int32).flatten()
                for c in range(self.n_class):
                    counts[c] += int((cls_labels == c).sum())
        eps = 1e-6
        inv_freq = 1.0 / (counts.astype(np.float64) + eps)
        inv_freq *= (self.n_class / inv_freq.sum()) if inv_freq.sum() > 0 else 1.0
        self.class_weights = inv_freq.astype(np.float32)

    def __len__(self) -> int:
        return sum(self.mesh_count)

    def __getitem__(self, index: int) -> dict:
        seg_id, in_seg_idx = _determine_segment(self.mesh_count, index)
        mesh_f = self.mesh_files[seg_id]
        label_f = self.labels_files[seg_id]
        if mesh_f not in self.opened_files:
            self.opened_files[mesh_f] = h5py.File(mesh_f, "r")
        if label_f not in self.opened_files:
            self.opened_files[label_f] = h5py.File(label_f, "r")
        hf_m = self.opened_files[mesh_f]
        hf_l = self.opened_files[label_f]

        v_flat = np.array(hf_m["verts"][in_seg_idx], dtype=np.float32)
        f_flat = np.array(hf_m["faces"][in_seg_idx], dtype=np.int32)
        v_shape = tuple(hf_m["vert_shapes"][in_seg_idx].tolist())
        f_shape = tuple(hf_m["face_shapes"][in_seg_idx].tolist())
        verts = v_flat.reshape(v_shape)
        faces = f_flat.reshape(f_shape)

        points = sample_points_uniformly(verts, faces, number_of_points=self.n_points, seed=self.seed)
        points_tensor = torch.tensor(normalize_pc(points, self.enable_rotate), dtype=torch.float32)
        points_np = points_tensor.numpy()
        L, M = point_cloud_laplacian(points_np)
        mass_np = M.diagonal().astype(np.float32).reshape(-1, 1)
        label = int(np.array(hf_l["cls_labels"][in_seg_idx], dtype=np.int32))
        return {
            "points": points_tensor,
            "mass": torch.tensor(mass_np, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "class_weights": torch.tensor(self.class_weights, dtype=torch.float32),
        }


class ClassificationMeshDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        n_class: int,
        enable_rotate: float = 0.0,
        labels_glob: str | None = None,
        mesh_glob: str | None = None,
    ) -> None:
        """Unified Dataset for Classification task using mesh Laplacian.

        Returns normalized vertices as `points`, along with `faces`, `mass`, and scalar `label`.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.n_class = n_class
        self.enable_rotate = enable_rotate
        mesh_pat = mesh_glob or "*_mesh.hdf5"
        labels_pat = labels_glob or "*_cls.hdf5"
        self.mesh_files = sorted([str(p) for p in self.data_dir.glob(mesh_pat)])
        self.labels_files = sorted([str(p) for p in self.data_dir.glob(labels_pat)])
        assert len(self.mesh_files) > 0
        assert len(self.mesh_files) == len(self.labels_files)
        self.mesh_count: list[int] = []
        for m, l in zip(self.mesh_files, self.labels_files):
            with h5py.File(m, "r") as hf_m, h5py.File(l, "r") as hf_l:
                count = len(hf_m["verts"])  # per exp preprocess format
                self.mesh_count.append(count)
                assert len(hf_l["cls_labels"]) == count
        self.opened_files: dict[str, h5py.File] = {}
        counts = np.zeros(self.n_class, dtype=np.int64)
        for lpath in self.labels_files:
            with h5py.File(lpath, "r") as hf_l:
                cls_labels = np.array(hf_l["cls_labels"], dtype=np.int32).flatten()
                for c in range(self.n_class):
                    counts[c] += int((cls_labels == c).sum())
        eps = 1e-6
        inv_freq = 1.0 / (counts.astype(np.float64) + eps)
        inv_freq *= (self.n_class / inv_freq.sum()) if inv_freq.sum() > 0 else 1.0
        self.class_weights = inv_freq.astype(np.float32)

    def __len__(self) -> int:
        return sum(self.mesh_count)

    def __getitem__(self, index: int) -> dict:
        seg_id, in_seg_idx = _determine_segment(self.mesh_count, index)
        mesh_f = self.mesh_files[seg_id]
        label_f = self.labels_files[seg_id]
        if mesh_f not in self.opened_files:
            self.opened_files[mesh_f] = h5py.File(mesh_f, "r")
        if label_f not in self.opened_files:
            self.opened_files[label_f] = h5py.File(label_f, "r")
        hf_m = self.opened_files[mesh_f]
        hf_l = self.opened_files[label_f]

        v_flat = np.array(hf_m["verts"][in_seg_idx], dtype=np.float32)
        f_flat = np.array(hf_m["faces"][in_seg_idx], dtype=np.int32)
        v_shape = tuple(hf_m["vert_shapes"][in_seg_idx].tolist())
        f_shape = tuple(hf_m["face_shapes"][in_seg_idx].tolist())
        verts = v_flat.reshape(v_shape)
        faces = f_flat.reshape(f_shape)

        L, M = mesh_laplacian(verts, faces)
        mass_np = M.diagonal().astype(np.float32).reshape(-1, 1)
        points_tensor = torch.tensor(normalize_pc(verts, self.enable_rotate), dtype=torch.float32)
        label = int(np.array(hf_l["cls_labels"][in_seg_idx], dtype=np.int32))
        return {
            "points": points_tensor,
            "faces": torch.tensor(faces, dtype=torch.long),
            "mass": torch.tensor(mass_np, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "class_weights": torch.tensor(self.class_weights, dtype=torch.float32),
        }


@dataclass
class UnifiedClsDataModuleConfig:
    batch_size: int
    data_dir: str
    n_class: int
    n_points: int = 1024
    use_mesh_laplacian: bool = False
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    limit_train: int = 0
    train_files: str = "*train_cls.hdf5"
    test_files: str = "*test_cls.hdf5"


class UnifiedClsDataModule(LightningDataModule):
    def __init__(self, cfg: UnifiedClsDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = self._infer_num_classes()
        if hasattr(self.cfg, "n_class") and self.cfg.n_class != self.num_classes:
            print(f"Auto-detected num_classes={self.num_classes}, overriding cfg.n_class={self.cfg.n_class}")
            self.cfg.n_class = self.num_classes

    def _make_dataset(self, split: str):
        labels_glob = self.cfg.train_files if split == "train" else self.cfg.test_files
        mesh_glob = labels_glob.replace("cls", "mesh")
        enable_rotate = self.cfg.enable_rotate if split == "train" else 0.0
        if self.cfg.use_mesh_laplacian:
            d = ClassificationMeshDataset(
                data_dir=self.cfg.data_dir,
                n_class=self.cfg.n_class,
                enable_rotate=enable_rotate,
                labels_glob=labels_glob,
                mesh_glob=mesh_glob,
            )
        else:
            d = ClassificationDataset(
                data_dir=self.cfg.data_dir,
                n_class=self.cfg.n_class,
                n_points=self.cfg.n_points,
                enable_rotate=enable_rotate,
                labels_glob=labels_glob,
                mesh_glob=mesh_glob,
            )
        print(f"Len of {split} dataset = {len(d)}")
        return d

    def _infer_num_classes(self) -> int:
        base = Path(self.cfg.data_dir)
        # prefer train; fallback to test
        candidates = list(sorted(base.glob(self.cfg.train_files)))
        if not candidates:
            candidates = list(sorted(base.glob(self.cfg.test_files)))
        if not candidates:
            raise RuntimeError(f"No classification label files found under {base} with patterns {self.cfg.train_files} / {self.cfg.test_files}")
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

    def train_dataloader(self):
        mp_ctx = "spawn" if self.cfg.num_workers > 0 else None
        dataset = self._make_dataset("train")
        if self.cfg.limit_train and self.cfg.limit_train > 0:
            from torch.utils.data import Subset
            if self.cfg.limit_train % self.cfg.n_class != 0:
                raise ValueError(f"limit_train must be divisible by n_class: {self.cfg.limit_train} % {self.cfg.n_class} != 0")
            quota = self.cfg.limit_train // self.cfg.n_class
            per_class_counts = [0 for _ in range(self.cfg.n_class)]
            indices: list[int] = []
            # Build indices by scanning cls_labels respecting dataset segment boundaries
            # Both ClassificationDataset and ClassificationMeshDataset expose labels_files and mesh_count
            mesh_counts = getattr(dataset, "mesh_count", None)
            labels_files = getattr(dataset, "labels_files", None)
            if mesh_counts is None or labels_files is None:
                raise RuntimeError("Dataset does not expose required fields for balanced limiting")
            offset = 0
            import h5py
            import numpy as np
            for seg_count, lbl_path in zip(mesh_counts, labels_files):
                with h5py.File(lbl_path, "r") as hf_l:
                    cls_labels = np.array(hf_l["cls_labels"][:seg_count], dtype=np.int32).flatten()
                for i in range(seg_count):
                    cls_id = int(cls_labels[i])
                    if 0 <= cls_id < self.cfg.n_class and per_class_counts[cls_id] < quota:
                        indices.append(offset + i)
                        per_class_counts[cls_id] += 1
                offset += seg_count
            # Ensure all classes reached quota
            if any(c < quota for c in per_class_counts):
                raise ValueError(f"Insufficient samples per class for balanced limit: got {per_class_counts}, require {quota} each")
            dataset = Subset(dataset, indices)
            print(f"Limiting train dataset to {len(dataset)} samples, quota per class = {quota}")
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
            batch_size=1,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
        )
