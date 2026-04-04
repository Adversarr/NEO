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
from g2pt.utils.gev import balance_stiffness
from g2pt.utils.mesh_feats import load_and_process_mesh, point_cloud_laplacian, sample_points_uniformly
from g2pt.utils.rot import random_rotate_3d
from g2pt.utils.sparse import to_torch_sparse_csr
import h5py

##### Data #####

@dataclass
class Geometry:
    """Pointcloud or Triangle mesh"""
    verts: np.ndarray
    faces: Optional[np.ndarray] = None

class BaseGeometryAdaptor:
    """Adaptor for pointcloud datasets.

    len: number of samples (shapes) in this dataset.
    get: get a sample (shape) from the dataset, and return it as a numpy array.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def len(self) -> int:
        """Get the total size of the dataset."""
        ...

    @abstractmethod
    def get(self, idx) -> Geometry:
        """Get a sample (shape) from the dataset.

        Args:
            idx (int): index of the sample.
            n_points (int): number of points to sample.

        Returns:
            Geometry: sampled geometry.
        """
        ...

class ShapenetAdaptor(BaseGeometryAdaptor):
    def __init__(self, base_path = '/dwata/ShapeNetCore.v2/') -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.cat_ids = sorted([d.name for d in self.base_path.iterdir() if d.is_dir()])
        self.samples = []
        for cat_id in self.cat_ids:
            cat_path = self.base_path / cat_id
            self.samples.extend([(cat_id, str(p)) for p in cat_path.glob("*")])
        print(f"Found {len(self.cat_ids)} categories and {len(self.samples)} samples.")

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx, n_points) -> Geometry:
        cat_id, path = self.samples[idx]
        verts, faces, mesh = load_and_process_mesh(path + "/models/model_normalized.obj")
        return Geometry(verts, faces)

class ProcessedShapenetAdaptor(BaseGeometryAdaptor):
    def __init__(self, base_path = '/data/processed_shapenet/') -> None:
        super().__init__()
        self.base_path = Path(base_path)
        self.ids = sorted([d.name for d in self.base_path.iterdir() if d.is_dir()])

        print(f"Found {len(self.ids)} samples.")

    def len(self) -> int:
        return len(self.ids)

    def get(self, idx) -> Geometry:
        path = self.base_path / self.ids[idx]
        verts = np.load(path / "trimesh_verts.npy")
        faces = np.load(path / "trimesh_faces.npy")
        return Geometry(verts, faces)


class UnifiedPreprocessedH5MeshAdaptor(BaseGeometryAdaptor):
    """
    Unified dataset for preprocessed HDF5 data from ShapeNet and Objaverse-Lowpoly.
    Scans for *_mesh.hdf5 and optional *_samples_evecs.hdf5, and serves
    points/evecs/mass either from precomputed samples or computed on the fly.
    """

    def __init__(
        self,
        data_dir: str,
    ):
        """
        Initialize the UnifiedPreprocessedH5Dataset.

        Args:
            data_dir: Directory containing the preprocessed HDF5 files
            enable_rotate: Probability of applying random rotation (0.0 to 1.0)
            targ_dim: Target dimension for eigenvectors
        """
        self.data_dir = Path(data_dir)
        # Find mesh and samples files (works for both ShapeNet and Objaverse-LP)
        self.mesh_files = sorted([str(path) for path in self.data_dir.glob("*_mesh.hdf5")])

        # Verify mesh files exist
        if not len(self.mesh_files) > 0:
            raise ValueError(f"No *_mesh.hdf5 files found in {data_dir}")
        # Load metadata from mesh files
        self.mesh_count = []
        for mesh_file in self.mesh_files:
            with h5py.File(mesh_file, "r") as hf:
                count = len(hf["verts"]) 
                self.mesh_count.append(count)
        self.opened_files: dict[str, h5py.File] = {}

    def len(self):
        return sum(self.mesh_count)

    def __del__(self):
        """Close all opened HDF5 files"""
        self.close_files()
    
    def close_files(self):
        """Close all opened HDF5 files"""
        if hasattr(self, "opened_files"):
            for f in self.opened_files.values():
                try:
                    f.close()
                except Exception as e:
                    print(f"Error closing file {f}: {e}")
            self.opened_files.clear()

    def _determine_which_segment(self, idx: int) -> tuple[int, int]:
        """Determine which mesh file (segment) and index within that file"""
        cum_count = 0
        for batch_idx, count in enumerate(self.mesh_count):
            if idx < cum_count + count:
                return batch_idx, idx - cum_count
            cum_count += count
        raise IndexError(f"Index {idx} out of bounds for dataset")
    
    def _ensure_file(self, file_path: str) -> h5py.File:
        if file_path not in self.opened_files:
            self.opened_files[file_path] = h5py.File(file_path, "r")
        return self.opened_files[file_path]

    def get(self, idx: int) -> Geometry:
        """Get a single sample from the dataset"""
        seg_id, in_seg_idx = self._determine_which_segment(idx)
        mesh_file = self._ensure_file(self.mesh_files[seg_id])
        
        v_shape = tuple(mesh_file["vert_shapes"][in_seg_idx].tolist())
        f_shape = tuple(mesh_file["face_shapes"][in_seg_idx].tolist())
        v_flat = np.array(mesh_file["verts"][in_seg_idx], dtype=np.float32)
        f_flat = np.array(mesh_file["faces"][in_seg_idx], dtype=np.int32)
        verts = v_flat.reshape(v_shape)
        faces = f_flat.reshape(f_shape)
        return Geometry(verts, faces)

def get_adaptor(name: str, base_path: str = '/data/ShapeNetCore.v2/') -> BaseGeometryAdaptor:
    if name == "shapenet":
        return ShapenetAdaptor(base_path)
    elif name == "processed_shapenet":
        return ProcessedShapenetAdaptor(base_path)
    elif name == 'unified_h5':
        return UnifiedPreprocessedH5MeshAdaptor(base_path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

class SelfSupervisedDataset(Dataset):
    def __init__(
        self,
        n_points: int,
        dataset_name: str,
        dataset_base_path: str,
        enable_rotate: float,
        target_k: int,
        delta: float = 1.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.adaptor = get_adaptor(dataset_name, dataset_base_path)
        self.enable_rotate = enable_rotate
        self.target_k = target_k
        self.delta = delta
        self.debug = debug

    def __len__(self) -> int:
        return self.adaptor.len()

    def __getitem__(self, idx) -> dict:
        geom = self.adaptor.get(idx)
        if self.debug:
            points = geom.verts[:self.n_points] # deterministic
        else:
            points = sample_points_uniformly(geom.verts, geom.faces, self.n_points)

        # Ensure bound by [-1, 1]
        mean_points = np.mean(points, axis=0, keepdims=True)
        points = points - mean_points # zero center
        if self.enable_rotate > 0:
            rot = random_rotate_3d(self.enable_rotate)
            points = points @ rot.T.astype(points.dtype)
        points = points / (np.max(np.abs(points)) + 1e-12) # normalize to [-1, 1]

        L, M = point_cloud_laplacian(points)
        L_b, M_b = balance_stiffness(L, M, self.delta, self.target_k)
        pc_tensor = torch.from_numpy(points)
        mass_balanced_tensor = torch.from_numpy(M_b.diagonal()).unsqueeze(-1)
        stiff_coo = sp.coo_matrix(L_b, copy=True)
        stiff_coo.sum_duplicates()

        result = {
            "points": pc_tensor.to(dtype=torch.float32),  # torch f32 dense [n_points, 3]
            "stiffness": stiff_coo,  # torch f32 sparse csr [npoints, npoints]
            "lumped_mass": mass_balanced_tensor.to(dtype=torch.float32),  # torch f32 dense [n_points, 1]
        }
        return result

def collate_fn(batch) -> dict[str, torch.Tensor]:
    """Collate function for SelfSupervisedDataset.

    Args:
        batch (list[dict[str, torch.Tensor]]): batch of samples.

    Returns:
        dict[str, torch.Tensor]: collated batch.
    """
    points = torch.stack([item["points"] for item in batch], dim=0) # (b, npoints, 3)
    lumped_mass = torch.stack([item["lumped_mass"] for item in batch], dim=0)     # (b, npoints, 1)
    # Gather stiffness matrices
    rows = []
    cols = []
    values = []
    npoints = points.shape[1]
    for i, item in enumerate(batch):
        coo: sp.coo_matrix = item['stiffness'] # [npoints, npoints]
        rows.append(coo.row + i * npoints)
        cols.append(coo.col + i * npoints)
        values.append(coo.data)

    larger_matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(npoints * len(batch), npoints * len(batch)),
        copy=True
    )
    csr_matrix = to_torch_sparse_csr(larger_matrix)

    stiff_indices = torch.from_numpy(np.vstack([larger_matrix.row, larger_matrix.col])).to(torch.long)
    stiff_values = torch.from_numpy(larger_matrix.data).to(torch.float32)

    result = {
        "points": points.to(dtype=torch.float32),
        # TODO: deprecate the coo based SPMV and requirement on pyg
        "stiff_indices": stiff_indices,
        "stiff_values": stiff_values,
        "stiff_csr": csr_matrix,
        "lumped_mass": lumped_mass.to(dtype=torch.float32),
    }
    return result

@dataclass
class SelfSupervisedDataModuleConfig:
    name: str
    data_dir: str
    n_points: int
    batch_size: int
    enable_rotate: float
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    split_ratio: float

class SelfSupervisedDataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.ds_cfg: SelfSupervisedDataModuleConfig = cfg.datamod
        self.batch_size = self.ds_cfg.batch_size
        self.target_k = cfg.targ_dim_model
        self.delta = cfg.balancing_delta
        self.debug = cfg.debug
        dataset = self._make_dataset('train')
        total = len(dataset)
        self.workers = self.ds_cfg.num_workers
        self.pin_memory = self.ds_cfg.pin_memory
        self.prefetch_factor = self.ds_cfg.prefetch_factor
        self.split_ratio = self.ds_cfg.split_ratio
        self.train_indices, self.val_indices = split(total, self.split_ratio, 1)
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _make_dataset(self, split: str):
        enable_rotate = self.ds_cfg.enable_rotate if split == 'train' else 0
        return SelfSupervisedDataset(
            n_points=self.ds_cfg.n_points,
            dataset_name=self.ds_cfg.name,
            dataset_base_path=self.ds_cfg.data_dir,
            enable_rotate=enable_rotate,
            target_k=self.target_k,
            delta=self.delta,
            debug=self.debug,
        )

    def train_dataloader(self):
        mp_ctx = 'spawn' if self.workers > 0 else None
        return DataLoader(
            Subset(self._make_dataset('train'), self.train_indices),
            batch_size=self.batch_size,
            multiprocessing_context=mp_ctx,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        mp_ctx = 'spawn' if self.workers > 0 else None
        return DataLoader(
            Subset(self._make_dataset('val'), self.val_indices),
            batch_size=self.batch_size,
            multiprocessing_context=mp_ctx,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

if __name__ == "__main__":
    dataset = SelfSupervisedDataset(
        n_points=1000,
        dataset_name="processed_shapenet",
        dataset_base_path="/data/processed_shapenet/",
        enable_rotate=0.,
        target_k=10,
        delta=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    print(batch["points"].shape)
    print(batch["stiffness"].shape)
    print(batch["mass"].shape)
    print(batch["lumped_mass"].shape)

    mass_matrix_diag = batch['lumped_mass'].flatten().numpy()
    stiffness = batch['stiffness'].to_dense().numpy()

    import scipy.linalg as la
    eigvals = la.eigvalsh(stiffness, np.diag(mass_matrix_diag))
    print(eigvals[:20]) # should range from [1, 10] approximately(have a good preconditioning)
