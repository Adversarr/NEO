from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

import h5py
from g2pt.data.transforms import normalize_pc

class UnifiedPreprocessedH5Dataset(Dataset):
    """
    Unified dataset for preprocessed HDF5 data from ShapeNet and Objaverse-Lowpoly.
    Scans for *_mesh.hdf5 and optional *_samples_evecs.hdf5, and serves
    points/evecs/mass either from precomputed samples or computed on the fly.
    """

    def __init__(
        self,
        data_dir: str,
        enable_rotate: float = 0.0,
        targ_dim: int = 96,
        recursive: bool = False,
        return_mesh: bool = False,
    ):
        """
        Initialize the UnifiedPreprocessedH5Dataset.

        Args:
            data_dir: Directory containing the preprocessed HDF5 files
            enable_rotate: Probability of applying random rotation (0.0 to 1.0)
            targ_dim: Target dimension for eigenvectors
            recursive: Whether to search for files recursively in subdirectories
        """
        self.data_dir = Path(data_dir)
        self.enable_rotate = enable_rotate
        self.targ_dim = targ_dim

        # Find mesh and samples files (works for both ShapeNet and Objaverse-LP)
        if recursive:
            self.mesh_files = []
            self.samples_evecs_files = []
            for subfolder in self.data_dir.glob("*"):
                self.mesh_files.extend([str(path) for path in subfolder.glob("*_mesh.hdf5")])
                self.samples_evecs_files.extend([str(path) for path in subfolder.glob("*_samples_evecs.hdf5")])
            self.mesh_files.sort()
            self.samples_evecs_files.sort()
            print(f"Found {len(self.mesh_files)} files (RECURSIVE)")
        else:
            self.mesh_files = sorted([str(path) for path in self.data_dir.glob("*_mesh.hdf5")])
            self.samples_evecs_files = sorted([str(path) for path in self.data_dir.glob("*_samples_evecs.hdf5")])
            print(f"Found {len(self.mesh_files)} files (NO RECURSIVE)")

        # Verify mesh files exist
        assert len(self.mesh_files) > 0, "No *_mesh.hdf5 files found in data_dir"
        # If samples_evecs files exist, they should match mesh files count
        if self.samples_evecs_files:
            assert len(self.mesh_files) == len(self.samples_evecs_files), "Number of mesh and samples_evecs files must match"

        # infer the metadata from files:
        found_per_mesh_counts = set()
        found_k = set()
        self.mesh_count = []
        # optional: collect obj_paths for debugging or mapping
        self.obj_paths = []

        # Load metadata from mesh files
        for mesh_file in self.mesh_files:
            with h5py.File(mesh_file, "r") as hf:
                count = len(hf["verts"]) 
                self.mesh_count.append(count)
                if "obj_paths" in hf:
                    obj_paths_batch = [path.decode('utf-8') for path in hf["obj_paths"]]
                    self.obj_paths.extend(obj_paths_batch)


        for samples_evec_file in self.samples_evecs_files:
            with h5py.File(samples_evec_file, "r") as hf:
                n_samples = hf.attrs["per_mesh_count"]
                found_per_mesh_counts.add(n_samples)
                k = hf.attrs["k"]
                found_k.add(k)

        self.per_mesh_count = min(found_per_mesh_counts)
        self.k = min(found_k)
        
        if self.k < self.targ_dim:
            raise ValueError(
                f"k={self.k} < {self.targ_dim}=targ_dim, which is not supported. "
                "Please decrease targ_dim or increase k at preprocess stage."
            )

        self.opened_files: dict[str, h5py.File] = {}
        self.return_mesh = return_mesh

    def __len__(self):
        if self.samples_evecs_files:
            # If we have precomputed samples/evecs, return total number of samples
            return sum(self.mesh_count) * self.per_mesh_count
        else:
            # Otherwise, return total number of meshes
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

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        # Use precomputed samples and eigenvectors
        mesh_idx = idx // self.per_mesh_count
        subsample_idx = idx % self.per_mesh_count
        seg_id, in_seg_idx = self._determine_which_segment(mesh_idx)
        samples_evec_file = self._ensure_file(self.samples_evecs_files[seg_id])

        points = np.array(samples_evec_file["samples"][in_seg_idx, subsample_idx, :, :], dtype=np.float32)
        evecs = np.array(samples_evec_file["evecs"][in_seg_idx, subsample_idx, :, : self.targ_dim], dtype=np.float32)
        mass = np.array(samples_evec_file["mass"][in_seg_idx, subsample_idx, :], dtype=np.float32)

        if self.return_mesh:
            mesh_file = self._ensure_file(self.mesh_files[seg_id])
            v_shape = tuple(mesh_file["vert_shapes"][in_seg_idx].tolist())
            f_shape = tuple(mesh_file["face_shapes"][in_seg_idx].tolist())
            v_flat = np.array(mesh_file["verts"][in_seg_idx], dtype=np.float32)
            f_flat = np.array(mesh_file["faces"][in_seg_idx], dtype=np.int32)
            verts = v_flat.reshape(v_shape)
            faces = f_flat.reshape(f_shape)
        
        result = {
            "points": torch.tensor(normalize_pc(points, self.enable_rotate), dtype=torch.float32),
            "evecs": torch.from_numpy(evecs).to(dtype=torch.float32),
            "mass": torch.from_numpy(mass).to(dtype=torch.float32).reshape(-1, 1),
        }
        
        if self.return_mesh:
            result["verts"] = torch.from_numpy(verts).to(dtype=torch.float32)
            result["faces"] = torch.from_numpy(faces).to(dtype=torch.int32)
            
        return result
