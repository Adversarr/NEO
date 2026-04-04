
import numpy as np
import scipy.sparse.linalg as spla
import trimesh
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from g2pt.utils.mesh_feats import mesh_laplacian

def main():
    mesh_path = "ldata/featuring_models/stanford-bunny.obj"
    print(f"Loading {mesh_path}...")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        print("Loaded Scene, concatenating geometry...")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    # Save the mesh to ensure consistency
    mesh.export("test_mesh.ply")
    print("Saved test_mesh.ply")
    
    print("Computing Laplacian...")
    L, M = mesh_laplacian(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    
    print("Computing Eigenvectors...")
    # Solve generalized eigenproblem L x = lambda M x
    # Sigma=-1e-3 to find smallest eigenvalues near 0
    # We use a negative sigma to make L - sigma*M = L + |sigma|*M positive definite
    vals, vecs = spla.eigsh(L, M=M, k=20, sigma=-1e-3, which='LM')
    
    print(f"Eigenvalues: {vals}")
    print(f"Eigenvectors shape: {vecs.shape}")
    
    np.save("test_basis.npy", vecs)
    print("Saved test_basis.npy")

if __name__ == "__main__":
    main()
