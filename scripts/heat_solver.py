#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import matplotlib as mpl
from pathlib import Path
from time import perf_counter
import sys

# Try to import from g2pt, assume it's in python path or src
try:
    from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian
except ImportError:
    # If running from scripts/ without package install, try adding ../src
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian

# -----------------------------
# Utils
# -----------------------------

def mean_edge_length(V, F):
    if F is None or len(F) == 0:
        return 0.0
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    L = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return float(L.mean())

def estimate_h_point_cloud(points, k=10):
    # Estimate h as average distance to k nearest neighbors
    # Using KDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)
    # dists[:, 0] is 0 (self), take mean of 1..k
    mean_dists = dists[:, 1:].mean()
    return float(mean_dists)

def _error_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(b), 1e-12)
    rel = abs_diff / denom

    b_abs = np.abs(b)
    rel_mask_thr = max(1e-12, 1e-3 * float(np.max(b_abs)))
    mask = b_abs >= rel_mask_thr
    if np.any(mask):
        rel_masked = abs_diff[mask] / np.maximum(b_abs[mask], 1e-12)
        masked_rel_l1_mean = float(np.mean(rel_masked))
        masked_rel_linf = float(np.max(rel_masked))
        masked_frac = float(np.mean(mask.astype(np.float64)))
    else:
        masked_rel_l1_mean = float("nan")
        masked_rel_linf = float("nan")
        masked_frac = 0.0

    return {
        "l1_mean": float(np.mean(abs_diff)),
        "l2_rmse": float(np.sqrt(np.mean(diff * diff))),
        "linf": float(np.max(abs_diff)),
        "rel_l1_mean": float(np.mean(rel)),
        "rel_linf": float(np.max(rel)),
        "rel_mask_threshold": float(rel_mask_thr),
        "rel_mask_fraction": float(masked_frac),
        "rel_masked_l1_mean": float(masked_rel_l1_mean),
        "rel_masked_linf": float(masked_rel_linf),
    }

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
    if span <= 0:
        d = np.zeros_like(values, dtype=np.float64)
    else:
        d = (values - lo) / span
    d = np.clip(d, 0.0, 1.0)

    cmap = mpl.colormaps[cmap_name]
    colors = cmap(d)
    colors = (colors[:, :3] * 255).astype(np.uint8)
    vertex_colors = np.hstack([colors, 255 * np.ones((len(colors), 1), dtype=np.uint8)])

    m = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=vertex_colors, process=False)
    m.vertex_attributes[feature_name] = values.astype(np.float32)
    m.export(str(out_path))

# -----------------------------
# Solver
# -----------------------------

class HeatDiffusionSolver:
    def __init__(self, points, faces=None, t=None, m=1.0):
        self.points = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64) if faces is not None else None
        self.n = self.points.shape[0]
        
        # Build Laplacian
        if self.faces is not None and len(self.faces) > 0:
            self.L, self.M = mesh_laplacian(self.points, self.faces)
            self.h = mean_edge_length(self.points, self.faces)
        else:
            self.L, self.M = point_cloud_laplacian(self.points)
            self.h = estimate_h_point_cloud(self.points)
            
        if t is None:
             t = float(m) * (self.h ** 2)
        self.t = float(t)
        self.m = float(m)
        
        # System: (M + tL) u = M u0
        self.A = (self.M + self.t * self.L).tocsr()
        self.solver = spla.factorized(self.A.tocsc())

    def solve_full(self, u0):
        u0 = np.asarray(u0, dtype=np.float64).reshape(-1)
        rhs = self.M @ u0
        u = self.solver(rhs)
        return u.ravel()
        
    def solve_subspace(self, u0, basis, ridge=1e-10):
        """
        Solve (M + tL) u = M u0 in subspace u = E alpha.
        (E^T (M + tL) E) alpha = E^T M u0
        """
        u0 = np.asarray(u0, dtype=np.float64).reshape(-1)
        E = np.asarray(basis, dtype=np.float64)
        k = E.shape[1]
        
        t0 = perf_counter()
        
        # Reduced system
        # A_red = E^T A E = E^T (M + tL) E
        AE = self.A @ E
        A_red = E.T @ AE
        
        # RHS
        rhs = self.M @ u0
        rhs_red = E.T @ rhs
        
        t_prep = perf_counter()
        
        # Regularize
        A_red = A_red + ridge * np.eye(k)
        
        alpha = np.linalg.solve(A_red, rhs_red)
        
        t_solve = perf_counter()
        
        u = E @ alpha
        
        t_recon = perf_counter()
        
        info = {
            "time_prep": t_prep - t0,
            "time_solve": t_solve - t_prep,
            "time_recon": t_recon - t_solve,
            "time_total": t_recon - t0,
            "k": k
        }
        return u.ravel(), info

# -----------------------------
# Data Loading
# -----------------------------

def _load_basis_from_infer_case(case_dir: Path, basis: str):
    input_dir = case_dir / "input"
    inferred_dir = case_dir / "inferred"

    # Try loading points
    if (input_dir / "points.npy").exists():
        points = np.load(input_dir / "points.npy").astype(np.float64)
    elif (case_dir / "sample_points.npy").exists():
        points = np.load(case_dir / "sample_points.npy").astype(np.float64)
    else:
        raise FileNotFoundError(f"Could not find points.npy or sample_points.npy in {case_dir}")

    # Try loading faces (optional)
    faces = None
    if (input_dir / "faces.npy").exists():
        faces = np.load(input_dir / "faces.npy").astype(np.int64)
    
    # Try loading mass (optional)
    mass = None
    if (input_dir / "mass.npy").exists():
        mass = np.load(input_dir / "mass.npy").astype(np.float64).reshape(-1)

    basis_evec = None
    basis_name = basis
    basis_file = ""

    if basis == "mesh":
        evec_path = inferred_dir / "mesh_evec.npy"
        if not evec_path.exists():
            # Fallback to input/mesh_evec.npy if not in inferred
            evec_path = input_dir / "mesh_evec.npy"
        
        if evec_path.exists():
            basis_evec = np.load(evec_path).astype(np.float64)
            basis_file = str(evec_path)
    elif basis == "net_fp32":
        evec_path = inferred_dir / "net_evec.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_evec_fp32.npy"
        if evec_path.exists():
            basis_evec = np.load(evec_path).astype(np.float64)
            basis_file = str(evec_path)
    elif basis == "pc_evec":
        evec_path = inferred_dir / "pc_evec.npy"
        if evec_path.exists():
            basis_evec = np.load(evec_path).astype(np.float64)
            basis_file = str(evec_path)
            
    # Fallback/General loader for explicit paths or other keys could go here
    if basis_evec is None:
         # Check if user provided a direct path in 'basis' argument (handled outside?)
         # Or maybe specific keys like 'net_pred_original'
         pass

    if basis_evec is None:
        raise ValueError(f"Could not load basis '{basis}' from {case_dir}")
        
    return points, faces, mass, basis_evec, basis_name, basis_file

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Heat Diffusion Solve: Full vs Subspace")
    parser.add_argument("--infer_case_dir", type=str, help="Path to inference case directory")
    parser.add_argument("--mesh", type=str, help="Path to mesh or point cloud file (if not using infer_case_dir)")
    parser.add_argument("--basis_path", type=str, help="Path to basis npy file (if not using infer_case_dir)")
    
    parser.add_argument("--basis", default="net_fp32", help="Basis type to load from infer_case_dir (mesh, net_fp32, pc_evec)")
    parser.add_argument("--basis_k", type=int, default=0, help="Truncate basis to k modes (0=all)")
    
    parser.add_argument("--source-vid", type=int, default=0, help="Source vertex index")
    parser.add_argument("--rhs_type", type=str, default="delta", choices=["delta", "randn", "fourier"], help="Type of RHS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--t", type=float, default=None, help="Time step t")
    parser.add_argument("--m", type=float, default=1.0, help="Time step multiplier t = m * h^2")
    
    parser.add_argument("--out_dir", type=str, default="output_heat", help="Output directory")
    parser.add_argument("--export_ply", action="store_true", help="Export PLY files")
    
    args = parser.parse_args()
    
    # 1. Load Geometry & Basis
    points = None
    faces = None
    basis = None
    basis_name = "custom"
    
    if args.infer_case_dir:
        case_dir = Path(args.infer_case_dir)
        points, faces, mass, basis, basis_name, basis_file = _load_basis_from_infer_case(case_dir, args.basis)
        print(f"Loaded case: {case_dir}")
    else:
        if not args.mesh:
            raise ValueError("Must provide --infer_case_dir or --mesh")
        
        mesh_path = Path(args.mesh)
        if mesh_path.suffix in ['.obj', '.ply', '.off', '.stl']:
            try:
                m = trimesh.load(mesh_path, process=False)
                if isinstance(m, trimesh.Scene):
                    print("Loaded Scene, concatenating geometry...")
                    m = trimesh.util.concatenate(tuple(m.geometry.values()))
                
                points = m.vertices
                faces = m.faces
            except Exception as e:
                print(f"Mesh load error: {e}, trying as point cloud or simple load...")
                # Might be point cloud PLY or other issue
                m = trimesh.load(mesh_path, process=False)
                if isinstance(m, trimesh.Scene):
                     m = trimesh.util.concatenate(tuple(m.geometry.values()))

                if hasattr(m, 'vertices'):
                    points = m.vertices
                    faces = m.faces if hasattr(m, 'faces') and len(m.faces) > 0 else None
        elif mesh_path.suffix == '.npy':
            points = np.load(mesh_path)
            faces = None
        
        if args.basis_path:
            basis = np.load(args.basis_path)
            basis_name = "custom_file"
    
    if points is None:
        raise ValueError("Failed to load geometry")
        
    if basis is not None and args.basis_k > 0:
        basis = basis[:, :args.basis_k]
        
    print(f"Geometry: {len(points)} points, {0 if faces is None else len(faces)} faces")
    if basis is not None:
        basis = basis.squeeze()
        print(f"Basis: {basis.shape}")
    
    # 2. Setup Solver
    if args.seed is not None:
        np.random.seed(args.seed)

    solver = HeatDiffusionSolver(points, faces, t=args.t, m=args.m)
    print(f"Solver setup: t={solver.t:.6g} (m={solver.m}, h={solver.h:.6g})")
    
    # 3. Define Heat Source
    u0 = np.zeros(len(points))
    if args.rhs_type == "delta":
        if args.source_vid < len(points):
            u0[args.source_vid] = 1.0
        else:
            print(f"Warning: source_vid {args.source_vid} out of range, using 0")
            u0[0] = 1.0
    elif args.rhs_type == "randn":
        u0 = np.random.randn(len(points))
    elif args.rhs_type == "fourier":
        # Random Fourier features
        # Center and normalize points for consistent frequency scaling
        p_cent = points - points.mean(axis=0)
        scale = np.max(np.abs(p_cent))
        if scale < 1e-8: scale = 1.0
        p_norm = p_cent / scale
        
        # Random frequencies (scale=5.0 gives some oscillation across the shape)
        n_feats = 10
        W = np.random.randn(3, n_feats) * 5.0 
        B = np.random.uniform(0, 2*np.pi, n_feats)
        
        features = np.cos(p_norm @ W + B)
        u0 = features.mean(axis=1)
        
    # 4. Full Solve
    print("Running full solve...")
    t0 = perf_counter()
    u_full = solver.solve_full(u0)
    t1 = perf_counter()
    print(f"Full solve done in {t1-t0:.4f}s")
    
    # 5. Subspace Solve
    u_sub = None
    if basis is not None:
        print("Running subspace solve...")
        u_sub, info = solver.solve_subspace(u0, basis)
        print(f"Subspace solve done in {info['time_total']:.4f}s")
        
        # Compare
        metrics = _error_metrics(u_sub, u_full)
        print("--- Metrics (Subspace vs Full) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.6g}")
            
    # 6. Export
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "t": solver.t,
        "m": solver.m,
        "h": solver.h,
        "source_vid": args.source_vid,
        "full_time": t1-t0,
    }
    
    if u_sub is not None:
        results["subspace_time"] = info["time_total"]
        results["metrics"] = metrics
        
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    if args.export_ply:
        export_ply_with_vertex_colors_and_feature(
            points, faces, u_full, out_dir / "heat_full.ply", feature_name="heat"
        )
        if u_sub is not None:
            export_ply_with_vertex_colors_and_feature(
                points, faces, u_sub, out_dir / "heat_sub.ply", feature_name="heat"
            )
            export_ply_with_vertex_colors_and_feature(
                points, faces, np.abs(u_full - u_sub), out_dir / "heat_error.ply", feature_name="error", cmap_name="magma"
            )
            
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
