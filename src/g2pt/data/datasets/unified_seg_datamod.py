from dataclasses import dataclass
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from g2pt.utils.mesh_feats import point_cloud_laplacian, mesh_laplacian
from g2pt.data.transforms import interpolate_labels, normalize_pc
from g2pt.data.common import determine_segment

def _determine_segment(mesh_count: list[int], idx: int) -> tuple[int, int]:
    return determine_segment(mesh_count, idx)


class SegmentationDataset(Dataset):
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
        """
        Unified Dataset for Segmentation task

        must contains:

        xx_mesh.hdf5:
            - vertices: (n_verts, 3)
            - faces: (n_faces, 3)
        xx_labels.hdf5:
            - labels: (n_verts, n_class)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.n_class = n_class
        mesh_pat = mesh_glob or "*_mesh.hdf5"
        labels_pat = labels_glob or "*_labels.hdf5"
        self.mesh_files = sorted([str(p) for p in self.data_dir.glob(mesh_pat)])
        self.labels_files = sorted([str(p) for p in self.data_dir.glob(labels_pat)])
        assert self.data_dir.is_dir(), f"{self.data_dir} is not a directory"
        assert len(self.mesh_files) > 0
        assert len(self.mesh_files) == len(self.labels_files)
        self.enable_rotate = enable_rotate
        self.mesh_count: list[int] = []
        for m, l in zip(self.mesh_files, self.labels_files):
            with h5py.File(m, "r") as hf_m, h5py.File(l, "r") as hf_l:
                count = len(hf_m["verts"])  # align with preprocess keys
                self.mesh_count.append(count)
                assert len(hf_l["labels"]) == count
        self.n_points = n_points
        self.opened_files: dict[str, h5py.File] = {}
        self.seed = seed
        # Precompute class weights across entire dataset
        counts = np.zeros(self.n_class, dtype=np.int64)
        for lpath in self.labels_files:
            with h5py.File(lpath, "r") as hf_l:
                labels_dset = hf_l["labels"]
                label_shapes = hf_l["label_shapes"] if "label_shapes" in hf_l.keys() else None
                seg_count = len(labels_dset)
                for i in range(seg_count):
                    lbl = np.array(labels_dset[i])
                    if label_shapes is not None:
                        lbl_len = int(np.array(label_shapes[i])[0])
                        lbl = lbl.reshape(lbl_len)
                    if lbl.ndim == 2:
                        # one-hot per-vertex
                        cls_idx = lbl.argmax(axis=1).astype(np.int64)
                    else:
                        cls_idx = lbl.astype(np.int64)
                    # accumulate
                    for c in range(self.n_class):
                        counts[c] += int((cls_idx == c).sum())
        eps = 1e-6
        total = float(counts.sum())
        inv_freq = 1.0 / (counts.astype(np.float64) + eps)
        # normalize to mean=1 to avoid scaling the overall loss
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
        labels_raw = np.array(hf_l["labels"][in_seg_idx])
        if "label_shapes" in hf_l.keys():
            lbl_len = int(np.array(hf_l["label_shapes"][in_seg_idx])[0])
            labels_raw = labels_raw.reshape(lbl_len)
        v_flat = np.array(hf_m["verts"][in_seg_idx], dtype=np.float32)
        f_flat = np.array(hf_m["faces"][in_seg_idx], dtype=np.int32)
        v_shape = tuple(hf_m["vert_shapes"][in_seg_idx].tolist())
        f_shape = tuple(hf_m["face_shapes"][in_seg_idx].tolist())
        verts = v_flat.reshape(v_shape)
        faces = f_flat.reshape(f_shape)
        if labels_raw.ndim == 1:
            labels_one_hot = np.eye(self.n_class, dtype=np.float32)[labels_raw.astype(np.int32)]
        else:
            labels_one_hot = labels_raw.astype(np.float32)
        points, labels = interpolate_labels(
            verts, faces, labels_one_hot, self.n_points, self.n_class, self.seed, hard=True
        )
        points_tensor = torch.tensor(normalize_pc(points, self.enable_rotate), dtype=torch.float32)
        points_np = points_tensor.numpy()
        L, M = point_cloud_laplacian(points_np)
        mass_np = M.diagonal().astype(np.float32).reshape(-1, 1)
        return {
            "points": points_tensor,
            "mass": torch.tensor(mass_np, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_weights": torch.tensor(self.class_weights, dtype=torch.float32),
        }


class SegmentationMeshDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        n_class: int,
        enable_rotate: float = 0.0,
        labels_glob: str | None = None,
        mesh_glob: str | None = None,
    ) -> None:
        """
        Unified Dataset for Segmentation task using mesh Laplacian

        requires paired files:
        *_mesh.hdf5: verts(vlen float32), faces(vlen int32), vert_shapes, face_shapes
        *_labels.hdf5: labels (n_verts, n_class)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.n_class = n_class
        mesh_pat = mesh_glob or "*_mesh.hdf5"
        labels_pat = labels_glob or "*_labels.hdf5"
        self.mesh_files = sorted([str(p) for p in self.data_dir.glob(mesh_pat)])
        self.labels_files = sorted([str(p) for p in self.data_dir.glob(labels_pat)])
        assert len(self.mesh_files) > 0
        assert len(self.mesh_files) == len(self.labels_files)
        self.enable_rotate = enable_rotate
        self.mesh_count: list[int] = []
        for m, l in zip(self.mesh_files, self.labels_files):
            with h5py.File(m, "r") as hf_m, h5py.File(l, "r") as hf_l:
                count = len(hf_m["verts"])  # per exp/shapenet/preprocess.py
                self.mesh_count.append(count)
                assert len(hf_l["labels"]) == count
        self.opened_files: dict[str, h5py.File] = {}
        # Precompute class weights based on face_labels if available, else vertex labels
        counts = np.zeros(self.n_class, dtype=np.int64)
        for lpath in self.labels_files:
            with h5py.File(lpath, "r") as hf_l:
                use_face = "face_labels" in hf_l.keys()
                seg_count = len(hf_l["labels"]) if not use_face else len(hf_l["face_labels"])
                for i in range(seg_count):
                    if use_face:
                        cls_idx = np.array(hf_l["face_labels"][i], dtype=np.int64).flatten()
                    else:
                        lbl = np.array(hf_l["labels"][i])
                        if "label_shapes" in hf_l.keys():
                            lbl_len = int(np.array(hf_l["label_shapes"][i])[0])
                            lbl = lbl.reshape(lbl_len)
                        if lbl.ndim == 2:
                            cls_idx = lbl.argmax(axis=1).astype(np.int64)
                        else:
                            cls_idx = lbl.astype(np.int64)
                    for c in range(self.n_class):
                        counts[c] += int((cls_idx == c).sum())
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

        labels_raw = np.array(hf_l["labels"][in_seg_idx])
        if "label_shapes" in hf_l.keys():
            lbl_len = int(np.array(hf_l["label_shapes"][in_seg_idx])[0])
            labels_raw = labels_raw.reshape(lbl_len)
        if labels_raw.ndim == 1:
            labels_np = labels_raw.astype(np.int64)
        else:
            labels_np = labels_raw.astype(np.float32)

        L, M = mesh_laplacian(verts, faces)
        mass_np = M.diagonal().astype(np.float32).reshape(-1, 1)
        points_tensor = torch.tensor(normalize_pc(verts, self.enable_rotate), dtype=torch.float32)
        face_labels = np.array(hf_l["face_labels"][in_seg_idx], dtype=np.int64).flatten()
        assert face_labels.shape[0] == faces.shape[0], "face_labels must have same length as faces"

        # If labels are provided one-hot per-vertex, harden to indices using argmax over classes per vertex
        if labels_np.ndim == 2:
            labels_np = labels_np.argmax(axis=1).astype(np.int64)
        return {
            "points": points_tensor,
            "labels": torch.tensor(labels_np, dtype=torch.long),  # (nV, )
            "mass": torch.tensor(mass_np, dtype=torch.float32),  # (nV, 1)
            "faces": torch.tensor(faces, dtype=torch.long),  # (nF, 3)
            "face_labels": torch.tensor(face_labels, dtype=torch.long),  # (nF, )
            "class_weights": torch.tensor(self.class_weights, dtype=torch.float32),
        }


@dataclass
class UnifiedSegDataModuleConfig:
    batch_size: int
    data_dir: str
    n_class: int = 8
    n_points: int = 1024
    use_mesh_laplacian: bool = False
    enable_rotate: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    limit_train: int = 0
    train_files: str = "*train_labels.hdf5"
    test_files: str = "*test_labels.hdf5"


class UnifiedSegDataModule(LightningDataModule):
    def __init__(self, cfg: UnifiedSegDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = self._infer_num_classes()
        if hasattr(self.cfg, "n_class") and self.cfg.n_class != self.num_classes:
            print(f"Auto-detected num_classes={self.num_classes}, overriding cfg.n_class={self.cfg.n_class}")
            self.cfg.n_class = self.num_classes

    def _make_dataset(self, split: str, force_use_mesh: bool = False):
        labels_glob = self.cfg.train_files if split == "train" else self.cfg.test_files
        mesh_glob = labels_glob.replace("labels", "mesh")
        enable_rotate = self.cfg.enable_rotate if split == "train" else 0.0
        if self.cfg.use_mesh_laplacian or force_use_mesh:
            print(f"Using mesh laplacian for {split} dataset")
            d = SegmentationMeshDataset(
                data_dir=self.cfg.data_dir,
                n_class=self.cfg.n_class,
                enable_rotate=enable_rotate,
                labels_glob=labels_glob,
                mesh_glob=mesh_glob,
            )
        else:
            d = SegmentationDataset(
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
        candidates = list(sorted(base.glob(self.cfg.train_files)))
        if not candidates:
            candidates = list(sorted(base.glob(self.cfg.test_files)))
        if not candidates:
            raise RuntimeError(f"No segmentation label files found under {base} with patterns {self.cfg.train_files} / {self.cfg.test_files}")
        first = candidates[0]
        with h5py.File(str(first), "r") as hf:
            if "num_classes" in hf.attrs:
                return int(hf.attrs["num_classes"]) 
            if "n_class" in hf.attrs:
                return int(hf.attrs["n_class"]) 
            if "labels" in hf.keys():
                try:
                    labels_sample = np.array(hf["labels"][0], dtype=np.int32)
                    if labels_sample.size == 0:
                        raise RuntimeError("empty labels sample")
                    return int(labels_sample.max() + 1)
                except Exception:
                    pass
            raise RuntimeError(f"Cannot infer num_classes from {first}; ensure 'num_classes' or 'n_class' attribute exists")

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
            batch_size=1,
            shuffle=False,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
        )
