"""SMPLX Deformation Dataset for reposing experiments."""

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from g2pt.utils.gev import balance_stiffness
from g2pt.utils.mesh_feats import mesh_laplacian
from g2pt.utils.rot import random_rotate_3d
from g2pt.utils.sparse import to_torch_sparse_csr

def normalize_pc2(pc: np.ndarray, pc2: np.ndarray, enable_rotate: float = 0.0):
    """
    Preprocess point cloud: center and normalize to unit sphere.

    Args:
        pc: Input point cloud array of shape (N, 3)
        enable_rotate: Probability of applying random rotation (0.0 to 1.0)

    Returns:
        Preprocessed point clouds pc, pc2, each of shape (N, 3)
    """
    
    pc = pc - np.mean(pc, axis=0, keepdims=True)
    pc2 = pc2 - np.mean(pc2, axis=0, keepdims=True)

    if enable_rotate > 0:
        rot = random_rotate_3d(enable_rotate)
        pc = pc @ rot.T.astype(pc.dtype)
        pc2 = pc2 @ rot.T.astype(pc2.dtype)

    max_abs = np.max(np.abs(pc))
    pc = pc / (max_abs + 1e-12)
    pc2 = pc2 / (max_abs + 1e-12)
    return pc, pc2


class SMPLXDeformationDataset(Dataset):
    """Dataset for SMPLX deformation tasks.
    
    Loads preprocessed HDF5 files containing source point clouds and their
    corresponding deformations to target poses. Returns vertex positions,
    output positions (deformed), and mass/Laplacian matrices.
    """

    def __init__(
        self,
        data_file: str,
        enable_rotate: float = 0.0,
        target_k: int = 128,
        delta: float = 1.0,
    ) -> None:
        """
        Initialize the SMPLXDeformationDataset.

        Args:
            data_file: Path to the HDF5 file containing preprocessed deformation data.
            enable_rotate: Probability of applying random rotation (0.0 to 1.0).
            target_k: Target dimension for eigenvectors (used for stiffness balancing).
            delta: Delta parameter for stiffness balancing.
        """
        self.data_file = Path(data_file)
        self.enable_rotate = enable_rotate
        self.target_k = target_k
        self.delta = delta

        # Open HDF5 file and load metadata
        self.hf = h5py.File(self.data_file, "r")
        self.num_samples = len(self.hf["samples"])
        self.n_points = self.hf["samples"].shape[1]

        # Load faces (required for mesh laplacian)
        assert "faces" in self.hf, "Faces are required for SMPLX deformation dataset."
        self.faces = np.array(self.hf["faces"], dtype=np.int32)

        # Load precomputed Laplacian structure if available
        if "stiff_row" in self.hf and "stiff_col" in self.hf:
            self.has_precomputed_laplacian = True
        else:
            self.has_precomputed_laplacian = False

        print(f"Loaded SMPLX deformation dataset from {data_file}")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Points per sample: {self.n_points}")
        print(f"  Faces: {self.faces.shape}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset.

        Returns:
            dict containing:
                - points: Source vertex positions (torch.Tensor)
                - output_pos: Target vertex positions after deformation (torch.Tensor)
                - stiffness: Laplacian matrix (scipy.sparse.coo_matrix)
                - lumped_mass: Mass matrix diagonal (torch.Tensor)
        """
        # Load source points and deformation
        points = np.array(self.hf["samples"][idx], dtype=np.float32)
        deformation = np.array(self.hf["deformations"][idx], dtype=np.float32)

        # Load poses if available
        tar_poses = np.array(self.hf["tar_poses"][idx], dtype=np.float32)
        src_poses = np.array(self.hf["src_poses"][idx], dtype=np.float32)
        tar_poses_tensor = torch.from_numpy(tar_poses).to(dtype=torch.float32)
        src_poses_tensor = torch.from_numpy(src_poses).to(dtype=torch.float32)
        # Default 'poses' for backward compatibility, usually target pose is what we want
        poses_tensor = tar_poses_tensor 

        # Compute output position (deformed points)
        output_pos = points + deformation

        # Note: Data is already normalized in preprocessing stage, so we don't normalize again
        # points, output_pos = normalize_pc2(points, output_pos, self.enable_rotate)

        # Compute or load Laplacian and mass matrices
        if self.has_precomputed_laplacian:
            stiff_row = np.array(self.hf["stiff_row"][idx], dtype=np.int32)
            stiff_col = np.array(self.hf["stiff_col"][idx], dtype=np.int32)
            stiff_values = np.array(self.hf["stiff_values"][idx], dtype=np.float32)
            lumped_mass_diag = np.array(self.hf["lumped_mass"][idx], dtype=np.float32)
            L = sp.coo_matrix((stiff_values, (stiff_row, stiff_col)), 
                             shape=(self.n_points, self.n_points))
            # Note: lumped_mass_diag is already the diagonal
        else:
            L, M = mesh_laplacian(points, self.faces)
            lumped_mass_diag = M.diagonal()

        # Convert to tensors
        points_tensor = torch.from_numpy(points).to(dtype=torch.float32)
        output_pos_tensor = torch.from_numpy(output_pos).to(dtype=torch.float32)
        mass_balanced_tensor = torch.from_numpy(lumped_mass_diag).unsqueeze(-1).to(dtype=torch.float32)

        # Convert stiffness matrix to COO format
        stiff_coo = sp.coo_matrix(L, copy=True)
        stiff_coo.sum_duplicates()

        result = {
            "points": points_tensor,  # torch f32 dense [n_points, 3]
            "output_pos": output_pos_tensor,  # torch f32 dense [n_points, 3]
            "stiffness": stiff_coo,  # torch f32 sparse coo [n_points, n_points]
            "lumped_mass": mass_balanced_tensor,  # torch f32 dense [n_points, 1]
            "poses": poses_tensor, # torch f32 dense [n_pose_dims]
            "tar_poses": tar_poses_tensor,
            "src_poses": src_poses_tensor,
            "faces": torch.from_numpy(self.faces).to(dtype=torch.long) # torch i64 dense [n_faces, 3]
        }

        return result

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "hf") and self.hf is not None:
            self.hf.close()

    def __del__(self):
        """Ensure file is closed when dataset is destroyed."""
        self.close()


def collate_fn(batch) -> dict[str, torch.Tensor]:
    """Collate function for SMPLXDeformationDataset.

    Args:
        batch (list[dict[str, torch.Tensor]]): batch of samples.

    Returns:
        dict[str, torch.Tensor]: collated batch.
    """
    points = torch.stack([item["points"] for item in batch], dim=0)  # (b, n_points, 3)
    output_pos = torch.stack([item["output_pos"] for item in batch], dim=0)  # (b, n_points, 3)
    lumped_mass = torch.stack([item["lumped_mass"] for item in batch], dim=0)  # (b, n_points, 1)
    poses = torch.stack([item["poses"] for item in batch], dim=0) # (b, n_pose_dims)
    src_poses = torch.stack([item["src_poses"] for item in batch], dim=0) # (b, n_pose_dims)
    tar_poses = torch.stack([item["tar_poses"] for item in batch], dim=0) # (b, n_pose_dims)

    # Stack stiffness matrices (sparse)
    # Convert scipy sparse to torch sparse
    rows = []
    cols = []
    values = []
    n_points = points.shape[1]
    for i, item in enumerate(batch):
        coo = item["stiffness"]
        rows.append(coo.row + i * n_points)
        cols.append(coo.col + i * n_points)
        values.append(coo.data)

    larger_matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_points * len(batch), n_points * len(batch)),
        copy=True,
    )
    csr_matrix = to_torch_sparse_csr(larger_matrix)

    stiff_indices = torch.from_numpy(np.vstack([larger_matrix.row, larger_matrix.col])).to(torch.long)
    stiff_values = torch.from_numpy(larger_matrix.data).to(torch.float32)

    result = {
        "points": points,
        "output_pos": output_pos,
        "stiff_indices": stiff_indices,
        "stiff_values": stiff_values,
        "stiff_csr": csr_matrix,
        "lumped_mass": lumped_mass,
        "poses": poses,
        "src_poses": src_poses,
        "tar_poses": tar_poses,
        "faces": batch[0]["faces"],
    }

    return result

if __name__ == "__main__":
    from tqdm import tqdm
    dataset = SMPLXDeformationDataset(
        data_file="/home/adversarr/Repo/g2pt/ldata/preprocessed_deform/train_full-lap.hdf5",
        enable_rotate=0,
    )
    
    all_deforms = []
    all_poses = []
    print("Calculating dataset statistics...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        # Deformation is target - source in normalized space
        # deform = item["output_pos"] - item["points"]
        # all_deforms.append(deform.numpy())
        all_poses.append(item["poses"].numpy())
    
    # all_deforms = np.stack(all_deforms, axis=0) # [N, num_points, 3]
    all_poses = np.stack(all_poses, axis=0) # [N, n_pose_dims]
    # mean_deform = np.mean(all_deforms, axis=(0, 1))
    # std_deform = np.std(all_deforms, axis=(0, 1))
    
    # print(f"Mean Deformation: {mean_deform}")
    # print(f"Std Deformation: {std_deform}")

    print(f"Mean Pose: {np.mean(all_poses, axis=0)}")
