
import math
from pathlib import Path
import numpy as np
import trimesh

def load_and_process_mesh(obj_path: str | Path, file_type: str = "obj") -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh]:
    """
    Load and process a mesh from an OBJ file path.
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        tuple: (vertices, faces, processed_mesh)
    """
    mesh: trimesh.Trimesh | trimesh.Scene = trimesh.load_mesh(obj_path, file_type=file_type) # type: ignore
    # mesh: trimesh.Trimesh | trimesh.Scene = trimesh.load(obj_path, file_type="obj") # type: ignore
    
    if isinstance(mesh, trimesh.Trimesh):
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

    elif isinstance(mesh, trimesh.Scene):
        # merge every part into one mesh
        verts = []
        faces = []
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in {obj_path}")
        for key in mesh.geometry:
            part = mesh.geometry[key]
            verts.append(np.array(part.vertices))
            faces.append(np.array(part.faces))
        verts = np.concatenate(verts, axis=0)
        faces = np.concatenate(faces, axis=0)

    # Check for invalid values in vertices (NaN, Inf)
    if not np.all(np.isfinite(verts)):
        raise ValueError(f"Invalid values found in vertices of {obj_path} (NaN, Inf)")

    # Check for invalid values in faces (NaN, Inf, negative, or non-integer values)
    if not np.all(np.isfinite(faces)) or not np.all(faces >= 0) or not np.all(faces == faces.astype(int)):
        raise ValueError(f"Invalid values found in faces of {obj_path} (NaN, Inf, negative indices, or non-integer values)")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    return verts, faces, mesh

def split(total: int, ratio: float, multiplier: int, seed: int = 42) -> tuple[list[int], list[int]]:
    n_train = math.ceil(total * ratio)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Select from indices
    train_indices = rng.choice(total, n_train, replace=False)
    val_indices = np.setdiff1d(np.arange(total), train_indices)

    train_indices = train_indices * multiplier
    val_indices = val_indices * multiplier

    all_training = [train_indices + i for i in range(multiplier)]
    all_validating = [val_indices + i for i in range(multiplier)]

    return np.concatenate(all_training).tolist(), np.concatenate(all_validating).tolist()

def determine_segment(counter: list[int], idx: int) -> tuple[int, int]:
    cum = 0
    for i, c in enumerate(counter):
        if idx < cum + c:
            return i, idx - cum
        cum += c
    raise IndexError("Index out of bounds")
