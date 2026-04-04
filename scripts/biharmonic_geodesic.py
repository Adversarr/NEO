#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biharmonic Geodesic Distance Computation
References:
- appl.md: Mathematical definition and application context.
- heat_solver.py: Data loading and solver structure.
- smoothing.py: Basis loading and spectral processing.
"""

import argparse
import json
import numpy as np
import scipy.sparse as sp
import trimesh
import matplotlib as mpl
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from g2pt.utils.ortho_operations import qr_orthogonalization_numpy
from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian

# -----------------------------
# Data Classes & Utils
# -----------------------------

def normalize_geometry(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    mean = np.mean(pts, axis=0, keepdims=True)
    pts = pts - mean
    scale = np.max(np.abs(pts)) + 1e-12
    return pts / scale, mean.reshape(-1), float(scale)

@dataclass(frozen=True)
class BasisData:
    evec: np.ndarray
    evals: np.ndarray | None
    name: str
    evec_file: str
    eval_file: str | None

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _as_2d_evec(evec: np.ndarray) -> np.ndarray:
    evec = np.asarray(evec)
    if evec.ndim == 3 and evec.shape[0] == 1:
        evec = evec[0]
    if evec.ndim != 2:
        raise ValueError(f"basis must be 2D (N,k), got shape={evec.shape}")
    return np.asarray(evec, dtype=np.float64)

def export_ply_with_vertex_colors_and_feature(
    points: np.ndarray,
    faces: np.ndarray | None,
    values: np.ndarray,
    out_path: str | Path,
    *,
    feature_name: str,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(points):
        raise ValueError(f"values length mismatch: {values.shape} vs nV {len(points)}")

    lo = float(values.min()) if vmin is None else float(vmin)
    hi = float(values.max()) if vmax is None else float(vmax)
    span = hi - lo
    
    # Normalize for color mapping
    if span <= 1e-12:
        d = np.zeros_like(values, dtype=np.float64)
    else:
        d = (values - lo) / span
    d = np.clip(d, 0.0, 1.0)

    # Apply colormap
    # Using matplotlib for colormaps
    try:
        cmap = mpl.colormaps[cmap_name]
    except AttributeError:
        cmap = mpl.cm.get_cmap(cmap_name)
        
    colors = cmap(d)
    colors = (colors[:, :3] * 255).astype(np.uint8)
    vertex_colors = np.hstack([colors, 255 * np.ones((len(colors), 1), dtype=np.uint8)])

    m = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=vertex_colors, process=False)
    # Trimesh doesn't natively support custom scalar fields in PLY in a standard way that all viewers read,
    # but we can try adding it to vertex_attributes if supported by the exporter.
    # Alternatively, the color is the main visual.
    m.vertex_attributes[feature_name] = values.astype(np.float32)
    m.export(str(out_path))


def export_ply_with_vertex_rgba(
    points: np.ndarray,
    faces: np.ndarray | None,
    rgba_u8: np.ndarray,
    out_path: str | Path,
):
    rgba = np.asarray(rgba_u8, dtype=np.uint8)
    if rgba.ndim != 2 or rgba.shape[1] != 4:
        raise ValueError(f"rgba_u8 must be (N,4), got {rgba.shape}")
    if rgba.shape[0] != len(points):
        raise ValueError(f"rgba length mismatch: {rgba.shape} vs nV {len(points)}")
    m = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=rgba, process=False)
    m.export(str(out_path))


def _summary_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    qs = np.quantile(v, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p10": float(qs[3]),
        "median": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        return float("nan")
    rx = np.empty_like(x, dtype=np.float64)
    ry = np.empty_like(y, dtype=np.float64)
    rx[np.argsort(x)] = np.arange(x.size, dtype=np.float64)
    ry[np.argsort(y)] = np.arange(y.size, dtype=np.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _make_isoline_rgba(
    dist: np.ndarray,
    *,
    n_lines: int = 20,
    width: float = 0.02,
    rgb_u8: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    lo = float(np.nanmin(d))
    hi = float(np.nanmax(d))
    span = hi - lo
    if span <= 1e-12 or n_lines <= 0:
        rgba = np.zeros((d.size, 4), dtype=np.uint8)
        return rgba
    t = (d - lo) / span
    t = np.clip(t, 0.0, 1.0)
    step = 1.0 / float(n_lines)
    phase = t / step
    frac = phase - np.floor(phase)
    mask = np.abs(frac - 0.5) < float(width) * 0.5
    rgba = np.zeros((d.size, 4), dtype=np.uint8)
    rgba[mask, 0] = int(rgb_u8[0])
    rgba[mask, 1] = int(rgb_u8[1])
    rgba[mask, 2] = int(rgb_u8[2])
    rgba[mask, 3] = 255
    return rgba

# -----------------------------
# Data Loading (Adapted from smoothing.py)
# -----------------------------

def _load_points_faces_mass_from_case(case_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    input_dir = case_dir / "input"
    points = None
    faces = None
    mass = None

    # Load points
    if (input_dir / "points.npy").exists():
        points = np.load(input_dir / "points.npy").astype(np.float64)
    elif (case_dir / "sample_points.npy").exists():
        points = np.load(case_dir / "sample_points.npy").astype(np.float64)
    else:
        raise FileNotFoundError(f"Could not find input/points.npy or sample_points.npy under {case_dir}")

    # Load faces
    if (input_dir / "faces.npy").exists():
        faces = np.load(input_dir / "faces.npy").astype(np.int64)

    # Load mass
    if (input_dir / "mass.npy").exists():
        mass = np.load(input_dir / "mass.npy").astype(np.float64).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape} from {case_dir}")
    
    return points, faces, mass

def _load_basis_from_case(case_dir: Path, basis: str) -> BasisData:
    input_dir = case_dir / "input"
    inferred_dir = case_dir / "inferred"

    def _load_pair(evec_path: Path, eval_path: Optional[Path], name: str) -> BasisData:
        evec = _as_2d_evec(np.load(evec_path))
        evals = None
        eval_file = None
        if eval_path is not None and eval_path.exists():
            evals = np.load(eval_path).astype(np.float64).reshape(-1)
            eval_file = str(eval_path.as_posix())
        return BasisData(
            evec=evec,
            evals=evals,
            name=name,
            evec_file=str(evec_path.as_posix()),
            eval_file=eval_file,
        )

    if basis == "mesh":
        evec_path = inferred_dir / "mesh_evec.npy"
        eval_path = inferred_dir / "mesh_eval.npy"
        if not evec_path.exists():
            evec_path = input_dir / "mesh_evec.npy"
        if not eval_path.exists():
            eval_path = input_dir / "mesh_eval.npy"
        if not evec_path.exists():
            # Fallback to loading from 'mesh' if precomputed not found?
            # For now assume they exist as per smoothing.py
            raise FileNotFoundError(f"mesh_evec.npy not found under {case_dir}/(input|inferred)")
        return _load_pair(evec_path, eval_path if eval_path.exists() else None, "mesh")

    if basis == "pc":
        evec_path = inferred_dir / "pc_evec.npy"
        eval_path = inferred_dir / "pc_eval.npy"
        if not evec_path.exists():
             raise FileNotFoundError(f"pc_evec.npy not found under {inferred_dir}")
        return _load_pair(evec_path, eval_path if eval_path.exists() else None, "pc")

    if basis in {"net_fp32", "net"}:
        evec_path = inferred_dir / "net_evec.npy"
        eval_path = inferred_dir / "net_eval.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_evec_fp32.npy"
        if not eval_path.exists():
            eval_path = inferred_dir / "net_eval_fp32.npy"
        if not evec_path.exists():
             raise FileNotFoundError(f"net_evec*.npy not found under {inferred_dir}")
        return _load_pair(evec_path, eval_path if eval_path.exists() else None, "net_fp32")

    # Add other cases as needed from smoothing.py
    
    raise ValueError(f"Unknown basis={basis!r}")

def _subspace_eigendecomp(evec: np.ndarray, points: np.ndarray, faces: Optional[np.ndarray], ridge: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues in the subspace spanned by evec if they are missing."""
    E = np.asarray(evec, dtype=np.float64)
    
    if faces is not None and faces.size > 0:
        if mesh_laplacian is None:
             raise RuntimeError("mesh_laplacian unavailable")
        L, M = mesh_laplacian(points, faces)
    else:
        if point_cloud_laplacian is None:
             raise RuntimeError("point_cloud_laplacian unavailable")
        L, M = point_cloud_laplacian(points)

    # TODO: qr (mass aware to E)
    M_diag = M.diagonal() # (n, )
    E = qr_orthogonalization_numpy(E, M_diag.reshape(-1, 1)) # (n, d)

    Lr = E.T @ (L @ E)
    Mr = E.T @ (M @ E)
    Lr = 0.5 * (Lr + Lr.T)
    Mr = 0.5 * (Mr + Mr.T)


    # Increase ridge for numerical stability if mass matrix is ill-conditioned
    # The error "leading minor ... is not positive definite" usually means Mr is not PD.
    # Since M is positive definite (mass matrix), E^T M E should be PD if E has full rank.
    # If E has linearly dependent columns (or close to), Mr becomes singular.
    # We add a larger epsilon or use a more robust solver path.
    Mr = Mr + float(ridge) * np.eye(Mr.shape[0], dtype=np.float64)

    # Solve generalized eigenproblem Lr x = lambda Mr x
    try:
        from scipy import linalg as sla
        # Try standard eigh first
        evals, U = sla.eigh(Lr, Mr)
    except Exception: # Catch LinAlgError and others
        # Fallback 1: Try with larger ridge
        try:
            print("Warning: Mass matrix singular, adding ridge to improve PDness.")
            Mr_robust = Mr + 1e-6 * np.eye(Mr.shape[0], dtype=np.float64)
            from scipy import linalg as sla
            evals, U = sla.eigh(Lr, Mr_robust)
        except Exception:
            # Fallback 2: Standard eigenvalue problem inv(Mr) @ Lr
            # This is less stable for symmetric systems but avoids Cholesky decomp of Mr inside eigh
            try:
                print("Warning: Mass matrix singular, using standard eigenvalue problem inv(Mr) @ Lr.")
                A = np.linalg.solve(Mr, Lr)
                evals, U = np.linalg.eig(A)
                evals = np.real(evals)
                U = np.real(U)
            except Exception:
                 # Fallback 3: Least squares solve if Mr is singular
                 print("Warning: Mass matrix singular, using least squares for eigendecomposition.")
                 A = np.linalg.lstsq(Mr, Lr, rcond=None)[0]
                 evals, U = np.linalg.eig(A)
                 evals = np.real(evals)
                 U = np.real(U)

    # Sort
    order = np.argsort(evals)
    evals = evals[order]
    U = U[:, order]
    
    # Compute Ritz vectors (approximated eigenvectors in the full space)
    # Phi = E @ U
    ritz_evecs = E @ U
    
    return np.maximum(evals, 0.0), ritz_evecs

# -----------------------------
# Distance Functions
# -----------------------------

def compute_biharmonic_distance(
    eigenvectors: np.ndarray, 
    eigenvalues: np.ndarray, 
    source_idx: int,
    skip_first: bool = True
) -> np.ndarray:
    """
    Computes Biharmonic Distance: d(x,y)^2 = sum (phi_k(x) - phi_k(y))^2 / lambda_k^2
    """
    # 1. Select valid modes
    # If skip_first is True, we skip index 0 (assuming it's the constant mode with lambda ~ 0)
    start_k = 1 if skip_first else 0
    
    valid_evecs = eigenvectors[:, start_k:] # [N, K']
    valid_evals = eigenvalues[start_k:]     # [K']
    
    # Check for zero eigenvalues to avoid division by zero
    valid_evals = np.maximum(valid_evals, 1e-8)
    
    # 2. Compute weights: 1 / lambda^2
    weights = 1.0 / (valid_evals ** 2) # [K']
    
    # 3. Compute difference squared
    # source_vec: [1, K']
    source_vec = valid_evecs[source_idx, :].reshape(1, -1)
    
    # diff: [N, K']
    diff = valid_evecs - source_vec
    diff_sq = diff ** 2
    
    # 4. Weighted sum
    # dist_sq: [N]
    dist_sq = np.sum(diff_sq * weights.reshape(1, -1), axis=1)
    
    # 5. Sqrt
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    
    return dist

def compute_euclidean_distance(points: np.ndarray, source_idx: int) -> np.ndarray:
    src = points[source_idx, :].reshape(1, 3)
    dists = np.linalg.norm(points - src, axis=1)
    return dists

def compute_graph_geodesic(points: np.ndarray, faces: Optional[np.ndarray], source_idx: int, k_neighbors: int = 10) -> np.ndarray:
    """
    Computes approximate geodesic distance using graph shortest path (Dijkstra).
    If faces provided, uses mesh graph.
    If points only, uses k-NN graph.
    """
    from scipy.sparse.csgraph import dijkstra
    
    N = len(points)
    
    if faces is not None and len(faces) > 0:
        # Build adjacency matrix from faces
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        # Compute edge lengths
        # We need a sparse matrix (N, N) where entry (i,j) is distance
        # To do this efficiently:
        
        # Sort edges to handle undirected
        edges = np.sort(edges, axis=1)
        # Unique edges
        edges = np.unique(edges, axis=0)
        
        # Compute lengths
        vecs = points[edges[:, 1]] - points[edges[:, 0]]
        lens = np.linalg.norm(vecs, axis=1)
        
        # Build symmetric sparse matrix
        graph = sp.coo_matrix((lens, (edges[:, 0], edges[:, 1])), shape=(N, N))
        graph = graph + graph.T
    else:
        # k-NN graph
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, idxs = tree.query(points, k=k_neighbors+1)
        
        # Build edges
        # idxs: [N, k+1], dists: [N, k+1]
        # column 0 is self
        row_inds = np.repeat(np.arange(N), k_neighbors)
        col_inds = idxs[:, 1:].flatten()
        data = dists[:, 1:].flatten()
        
        graph = sp.coo_matrix((data, (row_inds, col_inds)), shape=(N, N))
        graph = graph + graph.T # Make symmetric (approx)
        
    # Run Dijkstra
    # indices=source_idx, return_predecessors=False
    d = dijkstra(graph, directed=False, indices=source_idx)
    return d

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute Biharmonic Geodesic Distance")
    parser.add_argument("--infer_case_dir", type=str, required=True, help="Path to inference case directory")
    parser.add_argument("--basis", type=str, default="net_fp32", help="Basis type (mesh, net_fp32, pc)")
    parser.add_argument("--basis_k", type=int, default=96, help="Number of basis used for biharmonic distance")
    parser.add_argument("--source_vid", type=int, default=0, help="Source vertex index")
    parser.add_argument("--out_dir", type=str, default="output_biharmonic", help="Output directory")
    parser.add_argument("--export_ply", action="store_true", help="Export PLY files for visualization")
    parser.add_argument("--compute_geodesic", action="store_true", help="Compute graph-based geodesic for comparison (slow for large meshes)")
    parser.add_argument("--skip_first_eval", action="store_true", default=True, help="Skip the first eigenvalue (assuming constant mode)")
    parser.add_argument("--cmap_dist", type=str, default="viridis", help="Colormap for distance visualization")
    parser.add_argument("--cmap_isoline", type=str, default="RdBu", help="Colormap for isoline visualization")
    parser.add_argument("--baseline_scipy", action="store_true", help="Compute baseline using scipy shift-invert eigendecomposition")
    
    args = parser.parse_args()
    
    case_dir = Path(args.infer_case_dir)
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)
    
    print(f"Loading case: {case_dir}")
    
    # 1. Load Geometry
    points, faces, mass = _load_points_faces_mass_from_case(case_dir)
    print(f"Geometry: {len(points)} points, {0 if faces is None else len(faces)} faces")
    
    # Normalize geometry: mean=0, bbox=[-1, 1]
    print("Normalizing geometry (mean=0, bbox=[-1, 1])...")
    points, norm_mean, norm_scale = normalize_geometry(points)
    
    if args.source_vid >= len(points):
        print(f"Warning: source_vid {args.source_vid} out of bounds, setting to 0")
        args.source_vid = 0
        
    # 2. Load Basis
    basis_data = _load_basis_from_case(case_dir, args.basis)
    evec = basis_data.evec
    evals = basis_data.evals
    
    # Truncate basis
    k = args.basis_k
    if k > 0 and k < evec.shape[1]:
        evec = evec[:, :k]
        if evals is not None:
            evals = evals[:k]
    
    print(f"Basis loaded: {basis_data.name}, shape {evec.shape}")
    
    # 3. Force Recompute Eigenvalues (Rayleigh Quotient)
    # User instruction: Input eigenvalues are unreliable. Always recompute.
    print("Recomputing eigenvalues and refining basis from subspace (Rayleigh Quotient)...")
    evals, evec = _subspace_eigendecomp(evec, points, faces)
    
    # Truncate eigenvalues if needed (evec already truncated above)
    if evals.shape[0] > k:
        evals = evals[:k]
    
    # 4. Compute Biharmonic Distance
    print("Computing Biharmonic Distance...")
    t0 = perf_counter()
    dist_biharmonic = compute_biharmonic_distance(evec, evals, args.source_vid, skip_first=args.skip_first_eval)
    t1 = perf_counter()
    print(f"Biharmonic distance computed in {t1-t0:.4f}s")
    
    # Initialize results dict early for scipy baseline
    results = {
        "case": str(case_dir),
        "source_vid": args.source_vid,
        "basis": args.basis,
        "basis_k": k,
        "n_points": int(len(points)),
        "n_faces": int(0 if faces is None else len(faces)),
        "normalize_mean": [float(v) for v in norm_mean.tolist()],
        "normalize_scale": float(norm_scale),
        "time_biharmonic": t1 - t0,
    }

    # 4b. Compute Scipy Baseline (Optional)
    dist_biharmonic_scipy = None
    if args.baseline_scipy:
        print(f"Computing Scipy Baseline (shift-invert) for k={k}...")
        t_scipy_start = perf_counter()
        
        # Get Laplacian
        if faces is not None and faces.size > 0:
            if mesh_laplacian is None:
                raise RuntimeError("mesh_laplacian unavailable")
            L, M = mesh_laplacian(points, faces)
        else:
            if point_cloud_laplacian is None:
                raise RuntimeError("point_cloud_laplacian unavailable")
            L, M = point_cloud_laplacian(points)
            
        from scipy.sparse.linalg import eigsh
        # shift-invert mode for smallest eigenvalues
        # sigma=0 finds eigenvalues closest to 0
        eps = 1.0e-8
        evals_scipy, evec_scipy = eigsh(L + eps * M, k=k, M=M, sigma=eps, which='LM')
        
        # Sort
        idx = np.argsort(evals_scipy)
        evals_scipy = evals_scipy[idx]
        evec_scipy = evec_scipy[:, idx]
        
        t_scipy_end = perf_counter()
        print(f"Scipy baseline computed in {t_scipy_end - t_scipy_start:.4f}s")
        
        dist_biharmonic_scipy = compute_biharmonic_distance(evec_scipy, evals_scipy, args.source_vid, skip_first=args.skip_first_eval)
        
        # Compare errors
        diff = dist_biharmonic - dist_biharmonic_scipy
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        linf = float(np.max(np.abs(diff)))
        rel_err = np.linalg.norm(diff) / (np.linalg.norm(dist_biharmonic_scipy) + 1e-8)
        spearman = _spearman_corr(dist_biharmonic, dist_biharmonic_scipy)
        
        print(f"--- Comparison (Inferred vs Scipy) ---")
        print(f"MSE: {mse:.6e}")
        print(f"MAE: {mae:.6e}")
        print(f"Linf: {linf:.6e}")
        print(f"Relative Error: {rel_err:.6e}")
        print(f"Spearman Corr: {spearman:.6e}")
        print(f"--------------------------------------")
        
        results["error_mse"] = float(mse)
        results["error_mae"] = float(mae)
        results["error_linf"] = float(linf)
        results["error_rel"] = float(rel_err)
        results["error_spearman"] = float(spearman)
        results["time_scipy"] = t_scipy_end - t_scipy_start

    # 4. Compute Euclidean Distance (Baseline)
    dist_euclidean = compute_euclidean_distance(points, args.source_vid)
    
    # 5. Compute Graph Geodesic (Optional Ground Truth approximation)
    dist_geodesic = None
    if args.compute_geodesic:
        print("Computing Graph Geodesic (Dijkstra)...")
        t_geo = perf_counter()
        dist_geodesic = compute_graph_geodesic(points, faces, args.source_vid)
        print(f"Geodesic computed in {perf_counter() - t_geo:.4f}s")

    # 6. Save Results
    np.save(out_dir / "dist_biharmonic.npy", dist_biharmonic.astype(np.float32))
    if dist_biharmonic_scipy is not None:
        np.save(out_dir / "dist_biharmonic_scipy.npy", dist_biharmonic_scipy.astype(np.float32))
    np.save(out_dir / "dist_euclidean.npy", dist_euclidean.astype(np.float32))
    if dist_geodesic is not None:
        np.save(out_dir / "dist_geodesic.npy", dist_geodesic.astype(np.float32))

    results["stats"] = {
        "biharmonic": _summary_stats(dist_biharmonic),
        "biharmonic_scipy": (None if dist_biharmonic_scipy is None else _summary_stats(dist_biharmonic_scipy)),
        "euclidean": _summary_stats(dist_euclidean),
        "geodesic": (None if dist_geodesic is None else _summary_stats(dist_geodesic)),
    }

    abs_err = None
    rel_err_map = None
    log_rel_err_map = None
    if dist_biharmonic_scipy is not None:
        abs_err = np.abs(dist_biharmonic - dist_biharmonic_scipy)
        rel_err_map = abs_err / (np.abs(dist_biharmonic_scipy) + 1e-8)
        log_rel_err_map = np.log1p(rel_err_map)
        np.save(out_dir / "dist_error_abs.npy", abs_err.astype(np.float32))
        np.save(out_dir / "dist_error_rel.npy", rel_err_map.astype(np.float32))
        np.save(out_dir / "dist_error_logrel.npy", log_rel_err_map.astype(np.float32))
        results["stats"]["error_abs"] = _summary_stats(abs_err)
        results["stats"]["error_rel"] = _summary_stats(rel_err_map)
        results["stats"]["error_logrel"] = _summary_stats(log_rel_err_map)
        
    if args.export_ply:
        print("Exporting PLYs...")
        # For better visualization (Isolines), we can apply sin(alpha * dist)
        # But here we export raw distance, user can color in viewer or we color with basic map
        
        # Biharmonic (Inferred)
        export_ply_with_vertex_colors_and_feature(
            points, faces, dist_biharmonic, out_dir / "biharmonic.ply", 
            feature_name="biharmonic_dist", cmap_name=args.cmap_dist
        )
        
        # Biharmonic (Scipy Baseline)
        if dist_biharmonic_scipy is not None:
            export_ply_with_vertex_colors_and_feature(
                points, faces, dist_biharmonic_scipy, out_dir / "biharmonic_scipy.ply",
                feature_name="biharmonic_dist_scipy", cmap_name=args.cmap_dist
            )

        # Euclidean
        export_ply_with_vertex_colors_and_feature(
            points, faces, dist_euclidean, out_dir / "euclidean.ply",
            feature_name="euclidean_dist", cmap_name=args.cmap_dist
        )
        
        # Geodesic
        if dist_geodesic is not None:
            export_ply_with_vertex_colors_and_feature(
                points, faces, dist_geodesic, out_dir / "geodesic.ply",
                feature_name="geodesic_dist", cmap_name=args.cmap_dist
            )

        # Isoline visualization helper (Biharmonic)
        # Map distance to sin(freq * dist)
        freq = 50.0 # Adjust frequency
        isoline = np.sin(freq * dist_biharmonic)
        export_ply_with_vertex_colors_and_feature(
            points, faces, isoline, out_dir / "biharmonic_isoline.ply",
            feature_name="isoline", cmap_name=args.cmap_isoline
        )

        if dist_biharmonic_scipy is not None:
            export_ply_with_vertex_colors_and_feature(
                points,
                faces,
                rel_err_map,
                out_dir / "biharmonic_error_rel.ply",
                feature_name="biharmonic_error_rel",
                cmap_name="magma",
                vmin=0.0,
                vmax=float(np.quantile(rel_err_map, 0.99)),
            )
            export_ply_with_vertex_colors_and_feature(
                points,
                faces,
                log_rel_err_map,
                out_dir / "biharmonic_error_logrel.ply",
                feature_name="biharmonic_error_logrel",
                cmap_name="magma",
                vmin=0.0,
                vmax=float(np.quantile(log_rel_err_map, 0.99)),
            )

            if faces is not None and faces.size > 0:
                iso_gt = _make_isoline_rgba(dist_biharmonic_scipy, n_lines=24, width=0.03, rgb_u8=(0, 0, 0))
                iso_pred = _make_isoline_rgba(dist_biharmonic, n_lines=24, width=0.03, rgb_u8=(220, 20, 60))
                export_ply_with_vertex_rgba(points, faces, iso_gt, out_dir / "isoline_gt.ply")
                export_ply_with_vertex_rgba(points, faces, iso_pred, out_dir / "isoline_pred.ply")
        
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Done. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
