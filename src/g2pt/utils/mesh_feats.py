from pathlib import Path

import numpy as np
import open3d as o3d
from robust_laplacian import point_cloud_laplacian as pcl, mesh_laplacian as robust_mesh_laplacian
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import trimesh

from g2pt.data.common import load_and_process_mesh # TODO: backward compatible, remove in future.

def hks(evecs: np.ndarray, evals: np.ndarray, t_min: float = 1e-3, t_max: float = 100.0, count: int = 128):
    """
    Compute Heat Kernel Signature (HKS) for given eigenvalues and eigenvectors:
        HKS(i,t) = sum(φ_k(i)² * exp(-λ_k t))

    Args:
        evecs (np.ndarray): Eigenvectors of the Laplacian matrix.
        evals (np.ndarray): Eigenvalues of the Laplacian matrix.
        t_min (float): Minimum time parameter for HKS.
        t_max (float): Maximum time parameter for HKS.
        count (int): Number of time steps to compute.

    Returns:
        np.ndarray: HKS values for each point at each time step.
    """
    t = np.logspace(np.log10(t_min), np.log10(t_max), num=count)
    evec_sqr = evecs**2
    # if normalize:
    exp_terms = np.exp(-np.outer(evals, t))  # (num_eigen, num_t)
    # print(exp_terms)
    hks = evec_sqr @ exp_terms  # (n, num_t)
    return hks

def hks_autoscale(evecs: np.ndarray, evals: np.ndarray, count: int = 128):
    return hks(evecs, evals, t_min=1e-2, t_max=1.0, count=count)


def solve_gev(
    L: np.ndarray | csr_matrix,
    M: np.ndarray | csr_matrix,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compatibility wrapper: solves GEV via the standard interface in g2pt.utils.gev.
    Keeps legacy signature and return type intact.
    """
    # Lazy import to avoid potential circular imports
    from g2pt.utils.gev import solve_gev as _gev_solve
    # Support lumped M vector
    if isinstance(M, np.ndarray) and M.ndim == 1:
        M_use = sp.diags(M)
    else:
        M_use = M
    evals, evecs = _gev_solve(L=L, M=M_use, k=k)
    return evals, evecs

def _clamp_mass(mass: np.ndarray, smallest: float = 0.1) -> np.ndarray:
    """
    Clamp the mass values to avoid numerical issues.

    Refer to: https://github.com/IntelligentGeometry/NeLo/blob/master/utils.py#L247
    """
    mass = mass.astype(np.float64)  # higher accuracy to avoid numerical issues
    rel_mass = mass / (mass.mean() + 1e-8)
    rel_mass = np.maximum(rel_mass, smallest)
    return rel_mass


def point_cloud_laplacian(points: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    L, M = pcl(points, n_neighbors=20)  # limit to 20.
    L = csr_matrix(L, dtype=np.float64)
    M = sp.diags(_clamp_mass(M.diagonal()), dtype=np.float64)
    return L, csr_matrix(M)

def mesh_laplacian(vertices: np.ndarray, faces: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
    L, M = robust_mesh_laplacian(vertices, faces)
    L = csr_matrix(L, dtype=np.float64)
    M = sp.diags(_clamp_mass(M.diagonal()), dtype=np.float64)
    return L, csr_matrix(M)

def sample_points_uniformly(vertices, faces, number_of_points: int, seed: int | None = None, return_face_index: bool = False):
    """Sample points uniformly from a triangle mesh.

    Args:
        vertices (np.ndarray): Vertices of the mesh.
        faces (np.ndarray): Faces of the mesh.
        number_of_points (int): Number of points to sample.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.
        return_face_index (bool, optional): Whether to return the face indices of the sampled points. Defaults to False.

    Returns:
        np.ndarray: Sampled points.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    points, face_index = trimesh.sample.sample_surface(mesh, number_of_points, seed=seed)
    if return_face_index:
        return np.array(points), np.array(face_index)
    return np.array(points)

def sample_points_non_uniformly(
    vertices,
    faces,
    number_of_points: int,
    seed: int | None = None,
    return_face_index: bool = False,
    sigma: float | None = None,
):
    """Sample points non-uniformly from a triangle mesh to create a locally dense point cloud.
    
    This function selects a random center point on the mesh surface and weights the sampling probability
    of each face based on its distance to this center using a Gaussian distribution. This results in
    a point cloud that is denser near the center and sparser further away.

    Args:
        vertices (np.ndarray): Vertices of the mesh.
        faces (np.ndarray): Faces of the mesh.
        number_of_points (int): Number of points to sample.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.
        return_face_index (bool, optional): Whether to return the face indices of the sampled points. Defaults to False.
        sigma (float | None, optional): Scale of the dense region. If None, uses 25% of the bbox diagonal.

    Returns:
        np.ndarray: Sampled points.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if seed is not None:
        np.random.seed(seed)

    # 1. Pick a random center point on the surface to be the focus of density
    # We sample 1 point uniformly to serve as the center
    center_point, _ = trimesh.sample.sample_surface(mesh, 1, seed=seed)
    center_point = center_point[0]
    
    # 2. Compute distances from this center to all face centroids
    centroids = mesh.triangles_center
    distances = np.linalg.norm(centroids - center_point, axis=1)
    
    # 3. Define sigma (scale of the dense region)
    if sigma is None:
        # Heuristic: 25% of the mesh bounding box diagonal
        bounds = mesh.bounds
        if bounds is None: # Should not happen for valid mesh
             bbox_diag = 1.0
        else:
            bbox_diag = np.linalg.norm(bounds[1] - bounds[0])
        
        sigma = 0.25 * bbox_diag
        if sigma == 0:
            sigma = 1.0

    # 4. Compute weights: Area * Gaussian(dist)
    # We want density ~ Gaussian, so Weight/Area ~ Gaussian => Weight ~ Area * Gaussian
    gaussian_weights = np.exp(-0.5 * (distances / sigma)**2)
    face_weights = mesh.area_faces * gaussian_weights
    
    # 5. Sample using the computed weights
    points, face_index = trimesh.sample.sample_surface(mesh, number_of_points, face_weight=face_weights, seed=seed)
    
    if return_face_index:
        return np.array(points), np.array(face_index)
    return np.array(points)

def process_single_mesh(
    tri: "o3d.geometry.TriangleMesh",
    npoints: int,
    k: int,
    nsamples: int,
    output_path: Path
):
    # the first is the original mesh
    for j in range(nsamples):
        pc = tri.sample_points_uniformly(number_of_points=npoints)
        points = np.array(pc.points)
        # zeroing the points to center them around the origin
        points = points - np.mean(points, axis=0, keepdims=True)
        points = points / np.max(np.abs(points)) * 0.95
        L, M = point_cloud_laplacian(points)
        sp.save_npz(output_path / f"L_{j}.npz", L)
        # Compute the eigenvalues and eigenvectors (full)
        eigenvalues, eigenvectors = solve_gev(L, M, k=k + 1)  # k+1 to include the zero eigenvalue
        # downcast the accuracy for smaller file size
        evals = eigenvalues[: k + 1].astype(np.float32)  # since torch does not use float64
        evecs = eigenvectors[:, : k + 1].astype(np.float32)
        points = points.astype(np.float32)
        mass = M.diagonal().astype(np.float32)

        # Save the features
        np.save(output_path / f"eval_{j}.npy", evals)  # only store the meaningfuls
        np.save(output_path / f"evec_{j}.npy", evecs)  # only store the meaningfuls
        np.save(output_path / f"points_{j}.npy", points)
        np.save(output_path / f"mass_{j}.npy", mass)
