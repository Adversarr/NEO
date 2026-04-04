#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import matplotlib as mpl
import igl
from pathlib import Path
from time import perf_counter

from robust_laplacian import mesh_laplacian

# -----------------------------
# Geometry / FEM operators
# -----------------------------


def cotan(a, b):
    # cot(theta) = dot(a,b)/||cross(a,b)||
    cr = np.linalg.norm(np.cross(a, b), axis=1)
    # avoid division by 0 for degenerate triangles
    cr = np.maximum(cr, 1e-30)
    return np.einsum("ij,ij->i", a, b) / cr


def face_gradients(V, F, u):
    """
    Compute per-face gradient of piecewise linear function u on triangles.
    Returns grads: (m,3)
    Formula: grad u = sum_i u_i * grad phi_i, with
    grad phi_i = (N x e_i) / (2A)
    where e_i is edge opposite vertex i (CCW).
    """
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    # face normal (not unit) and area
    N = np.cross(v1 - v0, v2 - v0)  # (m,3), magnitude = 2A
    dblA = np.linalg.norm(N, axis=1)
    dblA = np.maximum(dblA, 1e-30)

    # Opposite edges (CCW):
    # Opp to v0 is e0 = v2 - v1
    # Opp to v1 is e1 = v0 - v2
    # Opp to v2 is e2 = v1 - v0
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # N_unit not needed; use N / (2A) via cross then divide by dblA
    grad_phi0 = (
        np.cross(N, e0) / (dblA[:, None] ** 2) * dblA[:, None]
    )  # simplifies to cross(N, e0)/(2A)^? let's do stable below
    # The above line is messy; do directly: cross(n_unit, e)/ (2A) with n_unit = N/dblA, 2A = dblA
    n_unit = N / dblA[:, None]
    grad_phi0 = np.cross(n_unit, e0) / dblA[:, None]
    grad_phi1 = np.cross(n_unit, e1) / dblA[:, None]
    grad_phi2 = np.cross(n_unit, e2) / dblA[:, None]

    uf0, uf1, uf2 = u[i0], u[i1], u[i2]
    grads = uf0[:, None] * grad_phi0 + uf1[:, None] * grad_phi1 + uf2[:, None] * grad_phi2
    return grads


def vertex_divergence_from_face_field(V, F, Xf):
    """
    Compute integrated divergence at vertices for a piecewise-constant per-face vector field Xf.
    Discrete formula equivalent to FEM: b = -G^T M_f X  (up to sign conventions).
    We'll implement common cotan-based integrated divergence:
      div(i) = 1/2 sum_{faces around i} (cot theta1 * (e1 · Xf) + cot theta2 * (e2 · Xf))
    where e1,e2 are edges from vertex i within that face, and theta1,theta2 are angles opposite those edges.
    Returns b (n,) float
    """
    n = len(V)
    b = np.zeros(n, dtype=np.float64)

    # For each face (a,b,c) in CCW order
    a, bidx, c = F[:, 0], F[:, 1], F[:, 2]
    va, vb, vc = V[a], V[bidx], V[c]

    # angles: at vertex a is between (vb-va) and (vc-va), etc.
    cot_a = cotan(vb - va, vc - va)
    cot_b = cotan(vc - vb, va - vb)
    cot_c = cotan(va - vc, vb - vc)

    # For vertex a in this face, the two edges in the face are eab=vb-va and eac=vc-va
    eab = vb - va
    eac = vc - va
    # Opposite angles to these edges: edge (a,b) opposite vertex c => cot_c, edge (a,c) opposite vertex b => cot_b
    # Contribution to div at a: 0.5*(cot_b*(eac·X) + cot_c*(eab·X)) with sign consistent with Crane et al.
    # We'll use +, later Poisson solve with C phi = b.
    Xa = Xf
    np.add.at(b, a, 0.5 * (cot_b * np.einsum("ij,ij->i", eac, Xa) + cot_c * np.einsum("ij,ij->i", eab, Xa)))

    # For vertex b: edges (bc)=vc-vb, (ba)=va-vb; opposite angles: edge(b,c) opp a => cot_a; edge(b,a) opp c => cot_c
    ebc = vc - vb
    eba = va - vb
    np.add.at(b, bidx, 0.5 * (cot_c * np.einsum("ij,ij->i", eba, Xa) + cot_a * np.einsum("ij,ij->i", ebc, Xa)))

    # For vertex c: edges (ca)=va-vc, (cb)=vb-vc; opposite angles: edge(c,a) opp b => cot_b; edge(c,b) opp a => cot_a
    eca = va - vc
    ecb = vb - vc
    np.add.at(b, c, 0.5 * (cot_a * np.einsum("ij,ij->i", ecb, Xa) + cot_b * np.einsum("ij,ij->i", eca, Xa)))

    return b


def mean_edge_length(V, F):
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    L = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return float(L.mean())


# -----------------------------
# Heat method core
# -----------------------------


class HeatGeodesicSolver:
    def __init__(self, V, F, t=None, m=1.0, h=None, use_cholesky_if_possible=True, eps_grad=1e-12, mollify_factor=1e-5):
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        self.n = self.V.shape[0]
        self.eps_grad = eps_grad

        # Use robust-laplacian to build operators
        self.C, self.M = mesh_laplacian(self.V, self.F, mollify_factor=mollify_factor)
        self.A = self.M.diagonal()

        if h is None:
            h = mean_edge_length(self.V, self.F)
        self.h = float(h)

        if t is None:
            t = float(m) * (self.h**2)
        self.t = float(t)
        self.m = float(m)

        # Systems:
        # Heat: (M + tC) u = rhs
        self.K_heat = (self.M + self.t * self.C).tocsr()

        # Poisson: C phi = b   (note: C has nullspace for closed surfaces)
        self.K_pois = self.C.tocsr()

        # Factorizations (optional; SciPy has no built-in sparse cholesky unless scikit-sparse)
        self.heat_solver = None
        self.pois_solver = None

        if use_cholesky_if_possible:
            # splu works for general sparse; for SPD can use factorized
            self.heat_solver = spla.factorized(self.K_heat.tocsc())

            # Poisson may be singular on closed surfaces; we handle by pinning one vertex at solve-time
            # so factorization depends on pin choice -> do in solve_poisson().
        else:
            self.heat_solver = None

    def solve_heat(self, u0, rhs_mode="mass"):
        """
        rhs_mode:
          - "mass": (M + tC) u = M u0   (u0 as vertex values)
          - "delta": (M + tC) u = delta (Kronecker at sources)
        """
        u0 = np.asarray(u0, dtype=np.float64).reshape(-1)
        assert u0.shape[0] == self.n

        if rhs_mode == "mass":
            rhs = self.M @ u0
        elif rhs_mode == "delta":
            rhs = u0
        else:
            raise ValueError("rhs_mode must be 'mass' or 'delta'")

        if self.heat_solver is not None:
            u = self.heat_solver(rhs)
        else:
            u = spla.spsolve(self.K_heat, rhs)
        return np.asarray(u).ravel()

    def compute_X(self, u):
        grads = face_gradients(self.V, self.F, u)  # (nf,3)
        norm = np.linalg.norm(grads, axis=1)
        norm = np.maximum(norm, self.eps_grad)
        Xf = -grads / norm[:, None]
        return Xf

    def solve_poisson_pinned(self, b, pin_idx=0, pin_value=0.0):
        """
        Solve C phi = b with a pinned vertex to remove nullspace.
        """
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        assert b.shape[0] == self.n

        # Modify system by pinning row/col
        C = self.K_pois.tolil()
        rhs = b.copy()

        C[pin_idx, :] = 0.0
        C[:, pin_idx] = 0.0
        C[pin_idx, pin_idx] = 1.0
        rhs[pin_idx] = pin_value

        C = C.tocsc()
        solver = spla.factorized(C)
        phi = solver(rhs)
        return np.asarray(phi).ravel()

    def solve_linear_system_under_basis(
        self,
        rhs: np.ndarray,
        basis: np.ndarray,
        A: sp.spmatrix,
        *,
        ridge: float = 1e-10,
    ) -> tuple[np.ndarray, dict]:
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs.shape[0] != self.n:
            raise ValueError(f"rhs length mismatch: rhs {rhs.shape} vs n {self.n}")

        E = np.asarray(basis, dtype=np.float64)
        if E.ndim != 2:
            raise ValueError(f"basis must be 2D, got {E.shape}")
        if E.shape[0] != self.n:
            raise ValueError(f"basis N mismatch: {E.shape} vs n {self.n}")
        k = int(E.shape[1])
        if k <= 0:
            raise ValueError(f"Empty basis: {E.shape}")

        info: dict[str, float | int] = {"k": int(k)}
        t0 = perf_counter()
        AE = A @ E
        t1 = perf_counter()
        A_red = E.T @ AE
        rhs_red = E.T @ rhs
        t2 = perf_counter()

        A_red = A_red + float(ridge) * np.eye(k, dtype=np.float64)
        solve0 = perf_counter()
        alpha = np.linalg.solve(A_red, rhs_red)
        solve1 = perf_counter()

        rec0 = perf_counter()
        x = E @ alpha
        rec1 = perf_counter()

        info.update(
            {
                "time_apply_A": float(t1 - t0),
                "time_reduce": float(t2 - t1),
                "time_solve": float(solve1 - solve0),
                "time_reconstruct": float(rec1 - rec0),
                "time_total": float(rec1 - t0),
            }
        )
        return np.asarray(x).ravel(), info

    def distance_single_source_under_subspace(
        self,
        source_vid: int,
        basis: np.ndarray,
        *,
        rhs_mode: str = "delta",
        basis_eval: np.ndarray | None = None,
        ridge: float = 1e-10,
        use_poisson_basis: bool = True,
    ) -> tuple[np.ndarray, dict]:
        s = int(source_vid)
        if not (0 <= s < self.n):
            raise ValueError(f"source_vid out of range: {s} for n={self.n}")
        if rhs_mode != "delta":
            raise ValueError("Only rhs_mode='delta' is supported in the subspace solver.")

        u0 = np.zeros(self.n, dtype=np.float64)
        u0[s] = 1.0

        u, heat_info = self.solve_linear_system_under_basis(rhs=u0, basis=basis, A=self.K_heat, ridge=ridge)
        heat_info["mode"] = "galerkin"
        heat_info["basis_eval_loaded"] = bool(basis_eval is not None)

        Xf = self.compute_X(u)
        b = vertex_divergence_from_face_field(self.V, self.F, Xf)

        if use_poisson_basis:
            phi, pois_info = self.solve_linear_system_under_basis(rhs=b, basis=basis, A=self.K_pois, ridge=ridge)
            pois_info["mode"] = "galerkin"
            pois_info["basis_eval_loaded"] = bool(basis_eval is not None)
            phi = phi - (phi[s] - 0.0)
        else:
            # Solve Poisson in full space
            t0 = perf_counter()
            phi = self.solve_poisson_pinned(b, pin_idx=s, pin_value=0.0)
            t1 = perf_counter()
            pois_info = {
                "mode": "full_space",
                "time_total": float(t1 - t0),
            }

        phi = phi - float(phi.min())
        return phi, {"mode": "galerkin", "heat": heat_info, "poisson": pois_info}


# -----------------------------
# CLI utilities
# -----------------------------


def parse_sources(mesh, args):
    if args.source_vid is not None and len(args.source_vid) > 0:
        return [int(x) for x in args.source_vid]

    if args.source_xyz is not None and len(args.source_xyz) == 3:
        p = np.array(args.source_xyz, dtype=np.float64)
        # nearest vertex
        d = np.linalg.norm(mesh.vertices - p[None, :], axis=1)
        vid = int(np.argmin(d))
        if args.k_nearest <= 1:
            return [vid]
        # take k nearest
        vids = np.argsort(d)[: args.k_nearest]
        return [int(x) for x in vids]

    raise ValueError("Please provide sources via --source-vid ... or --source-xyz x y z")


def export_ply_with_vertex_colors_and_feature(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    out_path: str | Path,
    *,
    feature_name: str,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(mesh.vertices):
        raise ValueError(f"values length mismatch: {values.shape} vs nV {len(mesh.vertices)}")

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

    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors, process=False)
    m.vertex_attributes[feature_name] = values.astype(np.float32)
    m.export(str(out_path))


def _load_basis_from_infer_mesh_case(
    case_dir: Path, basis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, str, str]:
    input_dir = case_dir / "input"
    inferred_dir = case_dir / "inferred"

    vertices = np.load(input_dir / "points.npy").astype(np.float64)
    faces = np.load(input_dir / "faces.npy").astype(np.int64)
    mass = np.load(input_dir / "mass.npy").astype(np.float64).reshape(-1)

    if basis == "mesh":
        evec = np.load(inferred_dir / "mesh_evec.npy").astype(np.float64)
        evals = np.load(inferred_dir / "mesh_eval.npy").astype(np.float64).reshape(-1)
        basis_name = "mesh_gt"
        basis_file = str((inferred_dir / "mesh_evec.npy").as_posix())
    elif basis == "net_fp16":
        evec_path = inferred_dir / "net_evec_fp16.npy"
        eval_path = inferred_dir / "net_eval_fp16.npy"
        evec = np.load(evec_path).astype(np.float64)
        evals = np.load(eval_path).astype(np.float64).reshape(-1)
        basis_name = "net_fp16"
        basis_file = str(evec_path.as_posix())
    elif basis == "net_fp32":
        evec_path = inferred_dir / "net_evec.npy"
        eval_path = inferred_dir / "net_eval.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_evec_fp32.npy"
        if not eval_path.exists():
            eval_path = inferred_dir / "net_eval_fp32.npy"
        evec = np.load(evec_path).astype(np.float64)
        evals = np.load(eval_path).astype(np.float64).reshape(-1)
        basis_name = "net_fp32"
        basis_file = str(evec_path.as_posix())
    elif basis in {"net_pred_fp32", "net_pred_original_fp32"}:
        evec_path = inferred_dir / "net_pred_original_fp32.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_pred_original.npy"
        evec = np.load(evec_path).astype(np.float64)
        if evec.ndim == 3 and evec.shape[0] == 1:
            evec = evec[0]
        if evec.ndim == 3 and evec.shape[-1] == 1:
            evec = evec[:, :, 0]
        evals = None
        basis_name = "net_pred_fp32"
        basis_file = str(evec_path.as_posix())
    elif basis in {"net_pred_fp16", "net_pred_original_fp16"}:
        evec_path = inferred_dir / "net_pred_original_fp16.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_pred_original.npy"
        evec = np.load(evec_path).astype(np.float64)
        if evec.ndim == 3 and evec.shape[0] == 1:
            evec = evec[0]
        if evec.ndim == 3 and evec.shape[-1] == 1:
            evec = evec[:, :, 0]
        evals = None
        basis_name = "net_pred_fp16"
        basis_file = str(evec_path.as_posix())
    else:
        raise ValueError(f"Unknown basis={basis!r}")

    if evec.ndim != 2:
        raise ValueError(f"Loaded basis must be 2D (N,D), got shape={evec.shape} from {basis_file}")

    return vertices, faces, mass, evec, evals, basis_name, basis_file


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


def main():
    ap = argparse.ArgumentParser(
        description="Heat Method geodesic distance on triangle meshes (Crane et al. TOG 2013)."
    )
    ap.add_argument("mesh", type=str, nargs="?", default=None, help="Input mesh file (obj/ply/stl/...).")
    ap.add_argument(
        "--infer_case_dir", type=str, default=None, help="Path to an infer_mesh.py case directory (OUTPUT/FILE1)."
    )
    ap.add_argument(
        "--basis",
        choices=[
            "net_fp32",
            "net_fp16",
            "mesh",
            "net_pred_fp32",
            "net_pred_fp16",
            "net_pred_original_fp32",
            "net_pred_original_fp16",
        ],
        default="net_fp32",
        help="Basis source when using --infer_case_dir.",
    )
    ap.add_argument("--basis_k", type=int, default=0, help="Number of basis modes to use (0 = all available).")
    ap.add_argument("--exp_out_dir", type=str, default=None, help="If set, write JSON + PLYs into this directory.")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-vid", type=int, nargs="+", help="Source vertex id(s). Example: --source-vid 10 20 30")
    src.add_argument(
        "--source-xyz", type=float, nargs=3, help="Source point in xyz; will snap to nearest vertex (or k nearest)."
    )

    ap.add_argument(
        "--k-nearest",
        type=int,
        default=1,
        help="If using --source-xyz, number of nearest vertices to use as multi-source (default 1).",
    )

    ap.add_argument(
        "--m",
        type=float,
        default=1.0,
        help="Time step multiplier: t = m*h^2 (default 1). Larger m => smoother distance.",
    )
    ap.add_argument("--t", type=float, default=None, help="Override time step t directly (if set, ignores --m).")

    ap.add_argument(
        "--rhs-mode",
        choices=["delta", "mass"],
        default="delta",
        help="Heat RHS: 'delta' uses Kronecker at sources; 'mass' uses M u0 (default delta).",
    )

    ap.add_argument(
        "--pin-idx", type=int, default=None, help="Pinned vertex index for Poisson (default: first source)."
    )

    ap.add_argument(
        "--eps-grad", type=float, default=1e-12, help="Epsilon for gradient norm in normalization (default 1e-12)."
    )
    ap.add_argument(
        "--mollify-factor", type=float, default=1e-5, help="Mollify factor for robust-laplacian (default 1e-5)."
    )
    ap.add_argument(
        "--no-poisson-basis",
        action="store_true",
        help="If set, do not use subspace basis for the Poisson solve step (use full space instead).",
    )

    ap.add_argument("--export-distance", type=str, default=None, help="Export distances to .npy file.")
    ap.add_argument(
        "--export-vertex-colors", type=str, default=None, help="Export colored mesh (PLY recommended). Example: out.ply"
    )
    ap.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib colormap for vertex colors (default: viridis)."
    )

    args = ap.parse_args()

    if args.infer_case_dir is not None:
        case_dir = Path(args.infer_case_dir)
        V, F, _mass_diag, basis_evec, basis_eval, basis_name, basis_file = _load_basis_from_infer_mesh_case(
            case_dir, args.basis
        )
        if args.basis_k > 0:
            k_use = int(min(args.basis_k, basis_evec.shape[1]))
            basis_evec = basis_evec[:, :k_use]
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        mesh_path_str = str((case_dir / "original.ply").as_posix())
    else:
        if args.mesh is None:
            raise ValueError("Need either positional mesh path or --infer_case_dir.")
        mesh = trimesh.load(args.mesh, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Input is not a single Trimesh. If it's a Scene, please export/merge to a single mesh.")
        V = mesh.vertices.view(np.ndarray)
        F = mesh.faces.view(np.ndarray)
        basis_evec = None
        basis_eval = None
        basis_name = "none"
        basis_file = ""
        mesh_path_str = str(Path(args.mesh).absolute().as_posix())

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces. This implementation needs triangles.")

    if not mesh.is_watertight:
        # Heat method works with boundaries too; just warning.
        pass

    source_vids = parse_sources(mesh, args)
    if len(source_vids) != 1:
        raise ValueError(f"This experimental script supports a single source only, got {len(source_vids)}.")
    source_vid = int(source_vids[0])

    solver = HeatGeodesicSolver(V, F, t=args.t, m=args.m, eps_grad=args.eps_grad, mollify_factor=args.mollify_factor)

    if args.rhs_mode != "delta":
        raise ValueError("This experimental script only supports --rhs-mode=delta.")

    # Compute reference exact geodesic using igl
    print(f"[igl] computing exact geodesic for {len(source_vids)} sources...")
    vs_indices = np.array(source_vids, dtype=np.int64)
    # igl.exact_geodesic(V, F, VS, FS, VT, FT)
    # Here we want distances to all vertices (VT = all vertices)
    vt_indices = np.arange(len(V), dtype=np.int64)
    dist_exact = igl.exact_geodesic(
        V, F, vs_indices, np.array([], dtype=np.int64), vt_indices, np.array([], dtype=np.int64)
    )

    full_u0 = np.zeros(len(V), dtype=np.float64)
    full_u0[source_vid] = 1.0
    t_full0 = perf_counter()
    t_heat0 = perf_counter()
    heat_u = solver.solve_heat(full_u0, rhs_mode="delta")
    t_heat1 = perf_counter()
    Xf_full = solver.compute_X(heat_u)
    b_full = vertex_divergence_from_face_field(V, F, Xf_full)
    t_pois0 = perf_counter()
    dist_full = solver.solve_poisson_pinned(b_full, pin_idx=source_vid, pin_value=0.0)
    t_pois1 = perf_counter()
    t_full1 = perf_counter()

    dist_full = dist_full - float(dist_full.min())
    dist_exact = dist_exact - float(dist_exact.min())

    print(f"[heat] nV={len(V)} nF={len(F)} h={solver.h:.6g} t={solver.t:.6g} (m={solver.m:.6g})")
    print(f"[src] source_vid={source_vid}")
    print(f"[dist_full] min={dist_full.min():.6g} max={dist_full.max():.6g}")
    print(f"[dist_exact] min={dist_exact.min():.6g} max={dist_exact.max():.6g}")

    # Compare heat vs exact
    diff = np.abs(dist_full - dist_exact)
    print(f"[compare full_vs_exact] L1_error={diff.mean():.6g} L_inf_error={diff.max():.6g}")

    if args.export_distance:
        np.save(args.export_distance, dist_full)
        print(f"[export] saved heat distances to {args.export_distance}")
        # Also save exact if requested? Let's just save heat as per original logic.

    if args.export_vertex_colors:
        # Use exact geodesic for colors as requested
        export_ply_with_vertex_colors_and_feature(
            mesh, dist_exact, args.export_vertex_colors, feature_name="dist_exact", cmap_name=args.cmap
        )
        print(f"[export] saved colored mesh (based on igl.exact_geodesic) to {args.export_vertex_colors}")

    if args.infer_case_dir is not None and args.exp_out_dir is not None:
        exp_out_dir = Path(args.exp_out_dir)
        exp_out_dir.mkdir(parents=True, exist_ok=True)

        t_sub0 = perf_counter()
        dist_sub, sub_info = solver.distance_single_source_under_subspace(
            source_vid=source_vid,
            basis=basis_evec,
            rhs_mode="delta",
            basis_eval=basis_eval,
            use_poisson_basis=not args.no_poisson_basis,
        )
        t_sub1 = perf_counter()

        dist_sub = dist_sub - float(dist_sub.min())

        err_sub_full = _error_metrics(dist_sub, dist_full)
        err_sub_exact = _error_metrics(dist_sub, dist_exact)

        summary = {
            "mesh": {
                "mesh_path": mesh_path_str,
                "n_verts": int(len(V)),
                "n_faces": int(len(F)),
            },
            "source": {
                "source_vid": int(source_vid),
            },
            "basis": {
                "basis": basis_name,
                "basis_file": basis_file,
                "k": int(basis_evec.shape[1]),
                "basis_eval_loaded": bool(basis_eval is not None),
                "basis_eval_used": False,
                "subspace_method": "project_solve_reconstruct",
            },
            "full_space_sparse": {
                "times": {
                    "heat_solve": float(t_heat1 - t_heat0),
                    "poisson_solve": float(t_pois1 - t_pois0),
                    "total": float(t_full1 - t_full0),
                }
            },
            "subspace_dense": {
                "times": {
                    "total": float(t_sub1 - t_sub0),
                },
                "details": sub_info,
            },
            "errors": {
                "subspace_vs_full": err_sub_full,
                "subspace_vs_exact": err_sub_exact,
                "full_vs_exact": _error_metrics(dist_full, dist_exact),
            },
        }
        with (exp_out_dir / "compare_solvers.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        dist_full = dist_full / dist_full.max()
        dist_sub = dist_sub / dist_sub.max()
        dist_exact = dist_exact / dist_exact.max()

        export_ply_with_vertex_colors_and_feature(
            mesh,
            dist_full,
            exp_out_dir / "dist_full_heat.ply",
            feature_name="dist_raw",
            cmap_name=args.cmap,
        )
        export_ply_with_vertex_colors_and_feature(
            mesh,
            dist_sub,
            exp_out_dir / f"dist_subspace_{basis_name}.ply",
            feature_name="dist_raw",
            cmap_name=args.cmap,
        )
        export_ply_with_vertex_colors_and_feature(
            mesh,
            dist_exact,
            exp_out_dir / "dist_exact_igl.ply",
            feature_name="dist_raw",
            cmap_name=args.cmap,
        )

        abs_err = np.abs(dist_sub - dist_full)
        export_ply_with_vertex_colors_and_feature(
            mesh,
            abs_err,
            exp_out_dir / f"abs_err_subspace_{basis_name}_vs_full.ply",
            feature_name="abs_err",
            cmap_name="magma",
        )

        print(f"[exp] wrote JSON + PLYs to {exp_out_dir}")


if __name__ == "__main__":
    main()
