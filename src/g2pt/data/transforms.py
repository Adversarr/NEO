import torch
from g2pt.utils.rot import random_rotate_3d
import numpy as np
from g2pt.utils.mesh_feats import sample_points_uniformly

def normalize_pc(pc: np.ndarray, enable_rotate: float = 0.0) -> np.ndarray:
    """
    Preprocess point cloud: center and normalize to unit sphere.

    Args:
        pc: Input point cloud array of shape (N, 3)
        enable_rotate: Probability of applying random rotation (0.0 to 1.0)

    Returns:
        Preprocessed point cloud array of shape (N, 3)
    """
    
    pc = pc - np.mean(pc, axis=0, keepdims=True)
    if enable_rotate > 0:
        rot = random_rotate_3d(enable_rotate)
        pc = pc @ rot.T.astype(pc.dtype)
    max_abs = np.max(np.abs(pc))
    pc = pc / (max_abs + 1e-12)
    return pc

def to_onehot(labels: np.ndarray, n_class: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        labels: Input label array of shape (N,) where N is the number of samples.
        n_class: Total number of classes.

    Returns:
        One-hot encoded label array of shape (N, n_class).
    """
    onehot = np.zeros((labels.shape[0], n_class), dtype=np.float32)
    labels = labels.astype(np.int64)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return onehot

def interpolate_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    num_points: int,
    num_classes: int,
    seed: int | None = None,
    hard: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly from a mesh and interpolate vertex labels using barycentric coordinates.
    
    Args:
        vertices: Vertex array of shape (N, 3) where N is the number of vertices.
        faces: Face index array of shape (M, 3) where M is the number of faces.
        labels: Label array of shape (N, C) where C is the number of classes, one-hot encoded.
                Note: Assumes per-vertex labels. If per-face labels are provided instead,
                the interpolation logic should be adjusted accordingly.
        num_points: Number of points to sample from the mesh surface.
        num_classes: Number of classes.
        seed: Random seed for reproducibility. If None, seed is not set.
    
    Returns:
        Sampled points of shape (num_points, 3) and labels (either indices (num_points,) if hard, or probabilities (num_points, C)).
    """
    # Sample points on mesh surface and get corresponding face indices
    # points shape: (num_points, 3), face_index shape: (num_points,)
    points, face_index = sample_points_uniformly(
        vertices,
        faces,
        number_of_points=num_points,
        seed=seed,
        return_face_index=True,
    )

    # Convert labels to one-hot encoding if not already
    if labels.ndim == 1:
        labels = to_onehot(labels, num_classes)
    elif labels.ndim == 2:
        if labels.shape[1] != num_classes:
            if labels.shape[1] == 1:
                labels = to_onehot(labels.flatten().astype(np.int64), num_classes)
            else:
                raise ValueError(f"Labels 2nd dimension {labels.shape[1]} does not match num_classes {num_classes}")
    else:
        raise ValueError(f"Labels must be 1D or 2D, got {labels.ndim}D")

    # Get the three vertex indices for each sampled point
    # vertex_indices shape: (num_points, 3)
    vertex_indices = faces[face_index]
    
    # Get the three vertices of each triangle
    # tri_verts shape: (num_points, 3, 3)
    tri_verts = vertices[vertex_indices]
    
    # Get the three labels corresponding to the triangle vertices
    # tri_labels shape: (num_points, 3, C)
    tri_labels = labels[vertex_indices]
    
    # Compute barycentric coordinates using area method
    # Unpack vertices for vectorized operations
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    
    # Compute total area of each face triangle
    # Using cross product magnitude; 0.5 factor is omitted as it cancels in ratios
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    area_total = np.linalg.norm(np.cross(v0v1, v0v2), axis=1)
    
    # Compute areas of sub-triangles formed by the sampled point
    area0 = np.linalg.norm(np.cross(v1 - points, v2 - points), axis=1)  # Opposite v0
    area1 = np.linalg.norm(np.cross(v2 - points, v0 - points), axis=1)  # Opposite v1
    area2 = np.linalg.norm(np.cross(v0 - points, v1 - points), axis=1)  # Opposite v2
    
    # Compute barycentric weights as area ratios
    # Add epsilon to prevent division by zero for degenerate triangles
    eps = 1e-10
    w0 = area0 / (area_total + eps)
    w1 = area1 / (area_total + eps)
    w2 = area2 / (area_total + eps)
    
    # Stack weights and normalize to ensure they sum to 1.0
    weights = np.stack([w0, w1, w2], axis=1)
    weights = np.clip(weights, 0.0, 1.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    # Interpolate labels using barycentric weights
    # weights[:, :, None] shape: (num_points, 3, 1)
    # tri_labels shape: (num_points, 3, C)
    # Result shape: (num_points, C)
    interpolated_labels = np.sum(weights[:, :, None] * tri_labels, axis=1)
    
    # Clip to maintain valid probability range for one-hot style labels
    interpolated_labels = np.clip(interpolated_labels, 0.0, 1.0)

    if hard:
        label_idx = np.argmax(interpolated_labels, axis=1).astype(np.int64)
        return points, label_idx
    else:
        return points, interpolated_labels.astype(np.float32)