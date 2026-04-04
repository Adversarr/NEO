"""
python /home/adversarr/Repo/g2pt/scripts/heat_distance.py --infer_case_dir tmp_mesh_infer_3/horse/ --source-vid 0 --basis net_fp32 --basis_k 20 --source-vid 0 --deflation-type additive 
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import trimesh
from robust_laplacian import mesh_laplacian, point_cloud_laplacian
import matplotlib.pyplot as plt



def cotan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cross_norm = np.linalg.norm(np.cross(a, b), axis=1)
    cross_norm = np.maximum(cross_norm, 1e-30)
    return np.einsum("ij,ij->i", a, b) / cross_norm


def face_gradients(vertices: np.ndarray, faces: np.ndarray, u: np.ndarray) -> np.ndarray:
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    normals = np.cross(v1 - v0, v2 - v0)
    dbl_area = np.linalg.norm(normals, axis=1)
    dbl_area = np.maximum(dbl_area, 1e-30)
    unit_normals = normals / dbl_area[:, None]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    grad_phi0 = np.cross(unit_normals, e0) / dbl_area[:, None]
    grad_phi1 = np.cross(unit_normals, e1) / dbl_area[:, None]
    grad_phi2 = np.cross(unit_normals, e2) / dbl_area[:, None]

    u0, u1, u2 = u[i0], u[i1], u[i2]
    return u0[:, None] * grad_phi0 + u1[:, None] * grad_phi1 + u2[:, None] * grad_phi2


def vertex_divergence_from_face_field(vertices: np.ndarray, faces: np.ndarray, face_field: np.ndarray) -> np.ndarray:
    n_verts = int(vertices.shape[0])
    div = np.zeros(n_verts, dtype=np.float64)

    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    va, vb, vc = vertices[a], vertices[b], vertices[c]

    cot_a = cotan(vb - va, vc - va)
    cot_b = cotan(vc - vb, va - vb)
    cot_c = cotan(va - vc, vb - vc)

    eab = vb - va
    eac = vc - va
    np.add.at(
        div,
        a,
        0.5
        * (
            cot_b * np.einsum("ij,ij->i", eac, face_field)
            + cot_c * np.einsum("ij,ij->i", eab, face_field)
        ),
    )

    ebc = vc - vb
    eba = va - vb
    np.add.at(
        div,
        b,
        0.5
        * (
            cot_c * np.einsum("ij,ij->i", eba, face_field)
            + cot_a * np.einsum("ij,ij->i", ebc, face_field)
        ),
    )

    eca = va - vc
    ecb = vb - vc
    np.add.at(
        div,
        c,
        0.5
        * (
            cot_a * np.einsum("ij,ij->i", ecb, face_field)
            + cot_b * np.einsum("ij,ij->i", eca, face_field)
        ),
    )

    return div


def mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(lengths.mean())


def estimate_h_point_cloud(points: np.ndarray, *, k: int = 10) -> float:
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=int(k) + 1)
    mean_dists = dists[:, 1:].mean()
    return float(mean_dists)


@dataclass(frozen=True)
class DeflationSpace:
    y: np.ndarray
    ay: np.ndarray
    e_inv: np.ndarray

    @staticmethod
    def build(a: sp.spmatrix, y: np.ndarray, *, ridge: float = 1e-10) -> "DeflationSpace":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"deflation basis must be 2D (N,k), got {y.shape}")
        if y.shape[1] <= 0:
            raise ValueError(f"deflation basis must have k>0, got {y.shape}")
        if y.shape[0] != a.shape[0]:
            raise ValueError(f"deflation basis N mismatch: {y.shape} vs A {a.shape}")
        
        # Precompute sparse-dense product
        ay = (a @ y).astype(np.float64)
        # Form the coarse matrix
        e = (y.T @ ay).astype(np.float64)
        # Regularize
        if ridge > 0:
            e[np.diag_indices(e.shape[0])] += float(ridge)
        
        # Invert explicitly (since m is small, typically < 64)
        try:
            e_inv = np.linalg.inv(e)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix if ridge didn't help enough
            e_inv = np.linalg.pinv(e)
            
        return DeflationSpace(y=y, ay=ay, e_inv=e_inv)

    def coarse_solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        # alpha = E^{-1} (Y^T b)
        yt_b = self.y.T @ b
        alpha = self.e_inv @ yt_b
        return (self.y @ alpha).reshape(-1)

    def project_left(self, v: np.ndarray) -> np.ndarray:
        # P_D^T v = v - (A Y) E^{-1} (Y^T v)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        yt_v = self.y.T @ v
        beta = self.e_inv @ yt_v
        correction = self.ay @ beta
        return v - correction.reshape(-1)

    def project_a_orth(self, v: np.ndarray) -> np.ndarray:
        # Projects v onto A-orthogonal complement of Y
        # v <- v - Y (E^-1 (Y^T A v))
        # Note: Y^T A v = (A Y)^T v = ay.T @ v
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        ayt_v = self.ay.T @ v
        beta = self.e_inv @ ayt_v
        correction = self.y @ beta
        return v - correction.reshape(-1)


def _make_jacobi_preconditioner(a: sp.spmatrix, *, eps: float = 1e-30):
    diag = np.asarray(a.diagonal(), dtype=np.float64).reshape(-1)
    inv = 1.0 / np.maximum(diag, float(eps))

    def apply(v: np.ndarray) -> np.ndarray:
        return inv * np.asarray(v, dtype=np.float64).reshape(-1)

    return apply


def _make_ichol_preconditioner(a: sp.spmatrix):
    # Use ILU as a proxy for Incomplete Cholesky since Scipy doesn't have native ICHOL.
    # For SPD matrices, ILU(0) is often a good alternative.
    try:
        import ilupp
        ichol = ilupp.ICholTPreconditioner(a.tocsc(), add_fill_in=100, threshold=0.1)
        # ilu = spla.spilu(
        #     a.tocsc(),
        #     drop_tol=1e-4,
        #     fill_factor=2.0,
        #     permc_spec="MMD_AT_PLUS_A",
        #     options={"SymmetricMode": True},
        # )
    except RuntimeError:
        # Fallback to Jacobi if factorization fails
        return _make_jacobi_preconditioner(a)

    def apply(v: np.ndarray) -> np.ndarray:
        # return ilu.solve(np.asarray(v, dtype=np.float64).reshape(-1))
        return ichol @ (np.asarray(v, dtype=np.float64).reshape(-1))

    return apply


def _make_two_level_preconditioner(base_prec, deflation: DeflationSpace):
    """
    Constructs a two-level additive preconditioner:
    P_new^{-1} = P_base^{-1} + Y E^{-1} Y^T
    
    This allows using standard PCG to achieve deflation effects without explicit projection steps.
    """
    def apply(v: np.ndarray) -> np.ndarray:
        # 1. Apply base preconditioner (high-frequency smoother)
        z_high = base_prec(v)
        # 2. Apply coarse correction (low-frequency solver)
        z_low = deflation.coarse_solve(v)
        # Additive combination
        return z_high + z_low
        
    return apply


def deflated_pcg(
    a: sp.spmatrix,
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner=None,
    deflation: DeflationSpace | None = None,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 2000,
) -> tuple[np.ndarray, dict]:
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(b.shape[0])
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
        if deflation is not None:
            x = deflation.coarse_solve(b)
    else:
        x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        if x.shape[0] != n:
            raise ValueError(f"x0 length mismatch: {x.shape} vs {b.shape}")

    if preconditioner is None:
        def preconditioner(v: np.ndarray) -> np.ndarray:
            return np.asarray(v, dtype=np.float64).reshape(-1)

    t0 = perf_counter()
    r = b - (a @ x)
    # Initial projection of residual (P_D r)
    if deflation is not None:
        r = deflation.project_left(r)

    b_norm = float(np.linalg.norm(b))
    r_norm = float(np.linalg.norm(r))
    stop_thr = float(atol) + float(tol) * (b_norm if b_norm > 0 else 1.0)
    if r_norm <= stop_thr:
        return x, {
            "converged": True,
            "num_iter": 0,
            "res_norm": r_norm,
            "b_norm": b_norm,
            "time_total": float(perf_counter() - t0),
        }

    z = preconditioner(r)
    # Project initial direction to be A-orthogonal (P_D^T z)
    if deflation is not None:
        z = deflation.project_a_orth(z)
    p = z.copy()
    rz_old = float(np.dot(r, z))

    converged = False
    it = 0

    for it in range(1, int(maxiter) + 1):
        ap = a @ p
        denom = float(np.dot(p, ap))
        if denom == 0.0:
            break

        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        
        # Prevent error accumulation by recomputing residual periodically
        if (it % 500) == 0:
            r = b - (a @ x)
            if deflation is not None:
                r = deflation.project_left(r)

        # Optimization: Skip re-projecting r (r = deflation.project_left(r))
        # Rely on p being A-orthogonal to keep r in the effective subspace.

        r_norm = float(np.linalg.norm(r))
        if r_norm <= stop_thr:
            converged = True
            break

        z = preconditioner(r)
        # Project preconditioned residual to maintain A-orthogonality of p
        if deflation is not None:
            z = deflation.project_a_orth(z)

        rz_new = float(np.dot(r, z))
        if rz_old == 0.0:
            break
        beta = rz_new / rz_old

        if (it % 1000) == 0:
            print(f"PCG iter {it}: r_norm = {r_norm:.2e}")

        p = z + beta * p
        # Optimization: Skip re-projecting p (p = deflation.project_a_orth(p))
        # Since z is projected (A-orth) and p_old was A-orth, linear combo is A-orth.
        rz_old = rz_new

    t1 = perf_counter()
    return x, {
        "converged": bool(converged),
        "num_iter": int(it if it is not None else 0),
        "res_norm": float(r_norm),
        "b_norm": float(b_norm),
        "rel_res": float(r_norm / b_norm) if b_norm > 0 else float("nan"),
        "time_total": float(t1 - t0),
    }


def _pin_spd_system(a: sp.spmatrix, b: np.ndarray, *, pin_idx: int, pin_value: float = 0.0) -> tuple[sp.csr_matrix, np.ndarray]:
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a_lil = a.tolil()
    rhs = b.copy()
    a_lil[pin_idx, :] = 0.0
    a_lil[:, pin_idx] = 0.0
    a_lil[pin_idx, pin_idx] = 1.0
    rhs[pin_idx] = float(pin_value)
    return a_lil.tocsr(), rhs


class HeatDistancePCGSolver:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        t: float | None = None,
        m: float = 1.0,
        eps_grad: float = 1e-12,
        mollify_factor: float = 1e-5,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n = int(self.vertices.shape[0])
        self.eps_grad = float(eps_grad)

        self.cotan_laplacian, self.mass = mesh_laplacian(
            self.vertices, self.faces, mollify_factor=float(mollify_factor)
        )
        self.h = mean_edge_length(self.vertices, self.faces)
        if t is None:
            t = float(m) * (self.h**2)
        self.t = float(t)
        self.m = float(m)

        self.a_heat = (self.mass + self.t * self.cotan_laplacian).tocsr()
        self.a_poisson = self.cotan_laplacian.tocsr()

    def _compute_face_unit_field(self, u: np.ndarray) -> np.ndarray:
        grads = face_gradients(self.vertices, self.faces, u)
        norms = np.linalg.norm(grads, axis=1)
        norms = np.maximum(norms, self.eps_grad)
        return -grads / norms[:, None]

    def solve_heat(
        self,
        rhs: np.ndarray,
        *,
        basis: np.ndarray | None = None,
        tol: float = 1e-8,
        maxiter: int = 2000,
        ridge: float = 1e-10,
        preconditioner: str = "jacobi",
        use_deflation_in_loop: bool = False,
        deflation_type: str = "projection",
    ) -> tuple[np.ndarray, dict]:
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs.shape[0] != self.n:
            raise ValueError(f"rhs length mismatch: {rhs.shape} vs n {self.n}")

        if preconditioner == "jacobi":
            p = _make_jacobi_preconditioner(self.a_heat)
        elif preconditioner == "ichol":
            p = _make_ichol_preconditioner(self.a_heat)
        elif preconditioner == "identity":
            p = None
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

        defl = None
        x0 = None
        if basis is not None:
            defl = DeflationSpace.build(self.a_heat, basis, ridge=ridge)
            # Always use coarse solve for initial guess if basis is available (Warm Start)
            x0 = defl.coarse_solve(rhs)

        # Configure PCG based on deflation type
        pcg_defl = None
        
        if basis is not None:
            if deflation_type == "additive":
                # Enhance preconditioner with coarse correction
                # Standard PCG loop (pcg_defl = None)
                if p is None:
                     # Identity base
                    def identity_prec(v): return np.asarray(v, dtype=np.float64).reshape(-1)
                    p = _make_two_level_preconditioner(identity_prec, defl)
                else:
                    p = _make_two_level_preconditioner(p, defl)
            elif deflation_type == "projection":
                # Use deflation logic inside PCG if requested
                if use_deflation_in_loop:
                    pcg_defl = defl
            else:
                raise ValueError(f"Unknown deflation_type: {deflation_type!r}")

        return deflated_pcg(
            self.a_heat, rhs, x0=x0, preconditioner=p, deflation=pcg_defl, tol=tol, maxiter=maxiter
        )

    def solve_poisson_pinned(
        self,
        rhs: np.ndarray,
        *,
        pin_idx: int,
        pin_value: float = 0.0,
        basis: np.ndarray | None = None,
        tol: float = 1e-8,
        maxiter: int = 4000,
        ridge: float = 1e-10,
        preconditioner: str = "jacobi",
        deflation_type: str = "projection",
    ) -> tuple[np.ndarray, dict]:
        pin_idx = int(pin_idx)
        if not (0 <= pin_idx < self.n):
            raise ValueError(f"pin_idx out of range: {pin_idx} for n={self.n}")

        a_pin, b_pin = _pin_spd_system(self.a_poisson, rhs, pin_idx=pin_idx, pin_value=float(pin_value))

        if preconditioner == "jacobi":
            p = _make_jacobi_preconditioner(a_pin)
        elif preconditioner == "ichol":
            p = _make_ichol_preconditioner(a_pin)
        elif preconditioner == "identity":
            p = None
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

        defl = None
        pcg_defl = None
        
        if basis is not None:
            defl = DeflationSpace.build(a_pin, basis, ridge=ridge)
            
            if deflation_type == "additive":
                if p is None:
                    def identity_prec(v): return np.asarray(v, dtype=np.float64).reshape(-1)
                    p = _make_two_level_preconditioner(identity_prec, defl)
                else:
                    p = _make_two_level_preconditioner(p, defl)
            elif deflation_type == "projection":
                pcg_defl = defl
            else:
                 raise ValueError(f"Unknown deflation_type: {deflation_type!r}")

        return deflated_pcg(a_pin, b_pin, preconditioner=p, deflation=pcg_defl, tol=tol, maxiter=maxiter)

    def distance_single_source(
        self,
        source_vid: int,
        *,
        basis: np.ndarray | None = None,
        tol: float = 1e-8,
        maxiter_heat: int = 2000,
        maxiter_poisson: int = 4000,
        ridge: float = 1e-10,
        preconditioner: str = "jacobi",
        deflation_type: str = "projection",
    ) -> tuple[np.ndarray, dict]:
        source_vid = int(source_vid)
        if not (0 <= source_vid < self.n):
            raise ValueError(f"source_vid out of range: {source_vid} for n={self.n}")

        rhs_heat = np.zeros(self.n, dtype=np.float64)
        rhs_heat[source_vid] = 1.0

        heat_u, heat_info = self.solve_heat(
            rhs_heat,
            basis=basis,
            tol=tol,
            maxiter=maxiter_heat,
            ridge=ridge,
            preconditioner=preconditioner,
            use_deflation_in_loop=False, # Heat step usually doesn't need full deflation loop
            deflation_type=deflation_type,
        )

        face_field = self._compute_face_unit_field(heat_u)
        rhs_pois = vertex_divergence_from_face_field(self.vertices, self.faces, face_field)

        phi, pois_info = self.solve_poisson_pinned(
            rhs_pois,
            pin_idx=source_vid,
            pin_value=0.0,
            basis=basis,
            tol=tol,
            maxiter=maxiter_poisson,
            ridge=ridge,
            preconditioner=preconditioner,
            deflation_type=deflation_type,
        )

        phi = phi - float(phi[source_vid])
        phi = phi - float(phi.min())
        return phi, {"heat": heat_info, "poisson": pois_info}


class HeatDistancePointCloudPCGSolver:
    def __init__(
        self,
        points: np.ndarray,
        *,
        t: float | None = None,
        m: float = 1.0,
        eps_log: float = 1e-30,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.n = int(self.points.shape[0])
        self.eps_log = float(eps_log)

        lap, mass = point_cloud_laplacian(self.points)
        self.laplacian = sp.csr_matrix(lap, dtype=np.float64)
        self.mass = sp.csr_matrix(mass, dtype=np.float64)

        self.h = estimate_h_point_cloud(self.points)
        if t is None:
            t = float(m) * (self.h**2)
        self.t = float(t)
        self.m = float(m)

        self.a_heat = (self.mass + self.t * self.laplacian).tocsr()

    def solve_heat(
        self,
        rhs: np.ndarray,
        *,
        basis: np.ndarray | None = None,
        tol: float = 1e-8,
        maxiter: int = 2000,
        ridge: float = 1e-10,
        preconditioner: str = "jacobi",
        deflation_type: str = "projection",
    ) -> tuple[np.ndarray, dict]:
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs.shape[0] != self.n:
            raise ValueError(f"rhs length mismatch: {rhs.shape} vs n {self.n}")

        if preconditioner == "jacobi":
            p = _make_jacobi_preconditioner(self.a_heat)
        elif preconditioner == "ichol":
            p = _make_ichol_preconditioner(self.a_heat)
        elif preconditioner == "identity":
            p = None
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

        defl = None
        x0 = None
        pcg_defl = None

        if basis is not None:
            defl = DeflationSpace.build(self.a_heat, basis, ridge=ridge)
            x0 = defl.coarse_solve(rhs)

            if deflation_type == "additive":
                if p is None:
                    def identity_prec(v):
                        return np.asarray(v, dtype=np.float64).reshape(-1)

                    p = _make_two_level_preconditioner(identity_prec, defl)
                else:
                    p = _make_two_level_preconditioner(p, defl)
            elif deflation_type == "projection":
                pcg_defl = defl
            else:
                raise ValueError(f"Unknown deflation_type: {deflation_type!r}")

        return deflated_pcg(self.a_heat, rhs, x0=x0, preconditioner=p, deflation=pcg_defl, tol=tol, maxiter=maxiter)

    def distance_single_source(
        self,
        source_vid: int,
        *,
        basis: np.ndarray | None = None,
        tol: float = 1e-8,
        maxiter_heat: int = 2000,
        ridge: float = 1e-10,
        preconditioner: str = "jacobi",
        deflation_type: str = "projection",
    ) -> tuple[np.ndarray, dict]:
        source_vid = int(source_vid)
        if not (0 <= source_vid < self.n):
            raise ValueError(f"source_vid out of range: {source_vid} for n={self.n}")

        rhs_heat = np.zeros(self.n, dtype=np.float64)
        rhs_heat[source_vid] = 1.0

        heat_u, heat_info = self.solve_heat(
            rhs_heat,
            basis=basis,
            tol=tol,
            maxiter=maxiter_heat,
            ridge=ridge,
            preconditioner=preconditioner,
            deflation_type=deflation_type,
        )

        logu = np.log(np.maximum(heat_u, self.eps_log))
        logu = logu - float(np.max(logu))
        dist = np.sqrt(np.maximum(-4.0 * self.t * logu, 0.0))
        dist = dist - float(dist[source_vid])
        dist = dist - float(dist.min())
        return dist, {"heat": heat_info}


def _load_basis_from_infer_case(
    case_dir: Path, basis: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, str, str]:
    input_dir = case_dir / "input"
    inferred_dir = case_dir / "inferred"

    if (input_dir / "points.npy").exists():
        points = np.load(input_dir / "points.npy").astype(np.float64)
    elif (case_dir / "sample_points.npy").exists():
        points = np.load(case_dir / "sample_points.npy").astype(np.float64)
    else:
        raise FileNotFoundError(f"Could not find input/points.npy or sample_points.npy in {case_dir}")

    faces = None
    if (input_dir / "faces.npy").exists():
        faces = np.load(input_dir / "faces.npy").astype(np.int64)

    mass = None
    if (input_dir / "mass.npy").exists():
        mass = np.load(input_dir / "mass.npy").astype(np.float64).reshape(-1)

    if basis == "mesh":
        evec_path = inferred_dir / "mesh_evec.npy"
        if not evec_path.exists():
            evec_path = input_dir / "mesh_evec.npy"
        evec = np.load(evec_path).astype(np.float64)
        basis_name = "mesh_gt"
        basis_file = str(evec_path.as_posix())
    elif basis == "pc":
        evec_path = inferred_dir / "pc_evec.npy"
        evec = np.load(evec_path).astype(np.float64)
        basis_name = "pc_gt"
        basis_file = str(evec_path.as_posix())
    elif basis == "net_fp16":
        evec_path = inferred_dir / "net_evec_fp16.npy"
        evec = np.load(evec_path).astype(np.float64)
        basis_name = "net_fp16"
        basis_file = str(evec_path.as_posix())
    elif basis == "net_fp32":
        evec_path = inferred_dir / "net_evec.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_evec_fp32.npy"
        evec = np.load(evec_path).astype(np.float64)
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
        basis_name = "net_pred_fp16"
        basis_file = str(evec_path.as_posix())
    else:
        raise ValueError(f"Unknown basis={basis!r}")

    if evec.ndim != 2:
        raise ValueError(f"Loaded basis must be 2D (N,k), got {evec.shape} from {basis_file}")

    if int(points.shape[0]) != int(evec.shape[0]):
        raise ValueError(f"N mismatch: points {points.shape} vs basis {evec.shape} from {basis_file}")

    return points, faces, mass, evec, basis_name, basis_file


def parse_sources_from_points(points: np.ndarray, args) -> list[int]:
    if args.source_vid is not None and len(args.source_vid) > 0:
        return [int(x) for x in args.source_vid]

    if args.source_xyz is not None and len(args.source_xyz) == 3:
        p = np.array(args.source_xyz, dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64)
        d = np.linalg.norm(pts - p[None, :], axis=1)
        vid = int(np.argmin(d))
        if args.k_nearest <= 1:
            return [vid]
        vids = np.argsort(d)[: args.k_nearest]
        return [int(x) for x in vids]

    raise ValueError("Please provide sources via --source-vid ... or --source-xyz x y z")


def parse_sources(mesh: trimesh.Trimesh, args) -> list[int]:
    if args.source_vid is not None and len(args.source_vid) > 0:
        return [int(x) for x in args.source_vid]

    if args.source_xyz is not None and len(args.source_xyz) == 3:
        p = np.array(args.source_xyz, dtype=np.float64)
        d = np.linalg.norm(mesh.vertices - p[None, :], axis=1)
        vid = int(np.argmin(d))
        if args.k_nearest <= 1:
            return [vid]
        vids = np.argsort(d)[: args.k_nearest]
        return [int(x) for x in vids]

    raise ValueError("Please provide sources via --source-vid ... or --source-xyz x y z")


def main() -> None:
    ap = argparse.ArgumentParser(description="Heat method distance with NEO-style deflated PCG.")
    ap.add_argument("mesh", type=str, nargs="?", default=None, help="Input mesh file (obj/ply/stl/...).")
    ap.add_argument(
        "--infer_case_dir", type=str, default=None, help="Path to an infer.py or infer_mesh.py case directory."
    )
    ap.add_argument(
        "--basis",
        choices=[
            "net_fp32",
            "net_fp16",
            "mesh",
            "pc",
            "net_pred_fp32",
            "net_pred_fp16",
            "net_pred_original_fp32",
            "net_pred_original_fp16",
        ],
        default="net_fp32",
    )
    ap.add_argument("--basis_path", type=str, default=None, help="Direct path to a (N,k) basis .npy file.")
    ap.add_argument("--basis_k", type=int, default=8, help="Number of basis modes to use (0 = all). Default: 8.")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-vid", type=int, nargs="+")
    src.add_argument("--source-xyz", type=float, nargs=3)
    ap.add_argument("--k-nearest", type=int, default=1)

    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--eps-grad", type=float, default=1e-12)
    ap.add_argument("--mollify-factor", type=float, default=1e-5)

    ap.add_argument("--tol", type=float, nargs="+", default=[1e-8])
    ap.add_argument("--maxiter-heat", type=int, default=100000)
    ap.add_argument("--maxiter-poisson", type=int, default=100000)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--preconditioner", choices=["jacobi", "ichol", "identity"], default="ichol")
    ap.add_argument("--deflation-type", choices=["projection", "additive"], default="additive", help="Strategy for deflation. 'projection' uses Nicolaides-type deflation in PCG loop. 'additive' uses two-level preconditioner.")
    ap.add_argument("--no-deflation", action="store_true")

    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--export-distance", type=str, default=None)
    ap.add_argument("--export-json", type=str, default=None)

    args = ap.parse_args()

    basis = None
    basis_name = "none"
    basis_file = ""
    mode = "mesh"

    if args.infer_case_dir is not None:
        case_dir = Path(args.infer_case_dir)
        v, f, _mass, basis, basis_name, basis_file = _load_basis_from_infer_case(case_dir, args.basis)
        if args.basis_k > 0 and basis is not None:
            basis = basis[:, : int(min(args.basis_k, basis.shape[1]))]
        if f is None or int(np.asarray(f).size) == 0:
            mode = "pointcloud"
            mesh = None
        else:
            mode = "mesh"
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    else:
        if args.mesh is None:
            raise ValueError("Need either positional mesh path or --infer_case_dir.")
        mesh_path = Path(args.mesh)
        if mesh_path.suffix.lower() == ".npy":
            v = np.load(mesh_path).astype(np.float64)
            f = None
            mesh = None
            mode = "pointcloud"
        else:
            loaded = trimesh.load(args.mesh, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Trimesh) and getattr(loaded, "faces", None) is not None and len(loaded.faces) > 0:
                mesh = loaded
                v = mesh.vertices.view(np.ndarray)
                f = mesh.faces.view(np.ndarray)
                mode = "mesh"
            else:
                if hasattr(loaded, "vertices"):
                    v = np.asarray(loaded.vertices, dtype=np.float64)
                    f = None
                    mesh = None
                    mode = "pointcloud"
                else:
                    raise ValueError("Input is neither a Trimesh nor a point cloud with vertices.")

        if args.basis_path is not None:
            basis = np.load(args.basis_path).astype(np.float64)
            basis_name = "path"
            basis_file = str(Path(args.basis_path).absolute().as_posix())
            if basis.ndim == 3 and basis.shape[0] == 1:
                basis = basis[0]
            if basis.ndim != 2:
                raise ValueError(f"Loaded basis must be 2D (N,k), got {basis.shape} from {basis_file}")
            if args.basis_k > 0:
                basis = basis[:, : int(min(args.basis_k, basis.shape[1]))]

    if args.no_deflation:
        basis = None
        basis_name = "none"
        basis_file = ""

    if mode == "mesh":
        if mesh is None:
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        source_vids = parse_sources(mesh, args)
    else:
        source_vids = parse_sources_from_points(v, args)
    if len(source_vids) != 1:
        raise ValueError(f"This script supports a single source only, got {len(source_vids)}.")
    source_vid = int(source_vids[0])

    if mode == "mesh":
        solver = HeatDistancePCGSolver(
            v,
            f,
            t=args.t,
            m=args.m,
            eps_grad=args.eps_grad,
            mollify_factor=args.mollify_factor,
        )
        solve_kwargs = dict(maxiter_heat=args.maxiter_heat, maxiter_poisson=args.maxiter_poisson)
    else:
        solver = HeatDistancePointCloudPCGSolver(v, t=args.t, m=args.m)
        solve_kwargs = dict(maxiter_heat=args.maxiter_heat)

    tols = args.tol
    if not isinstance(tols, list):
        tols = [tols]

    # Find the smallest tolerance (highest precision) for PLY export
    min_tol = min(tols)
    all_summaries = []

    for tol in tols:
        print(f"🚀 Start {tol:.3e}...")
        t0 = perf_counter()
        dist, info = solver.distance_single_source(
            source_vid,
            basis=basis,
            tol=tol,
            **solve_kwargs,
            ridge=args.ridge,
            preconditioner=args.preconditioner,
            deflation_type=args.deflation_type,
        )
        t1 = perf_counter()

        summary = {
            "mode": str(mode),
            "mesh": {"n_verts": int(v.shape[0]), "n_faces": int(0 if f is None else f.shape[0])},
            "source": {"source_vid": int(source_vid)},
            "params": {
                "t": float(solver.t),
                "m": float(solver.m),
                "h": float(solver.h),
                "tol": float(tol),
            },
            "basis": {
                "basis": basis_name,
                "basis_file": basis_file,
                "k": int(0 if basis is None else basis.shape[1]),
            },
            "pcg": info,
            "time_total": float(t1 - t0),
            "dist": {"min": float(dist.min()), "max": float(dist.max())},
        }
        all_summaries.append(summary)

        # Only export PLY if this is the highest precision run (smallest tol)
        if args.export_distance is not None and tol == min_tol:
            p = Path(args.export_distance)
            # Use exact filename provided by user, no suffix
            out_path = p
            dist_max = dist.max()
            def scal(x):
                return x ** 1.5
            dist = scal(dist / dist_max)
            c = plt.get_cmap(args.cmap)(dist)[:, :3]
            m_export = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)
            m_export.export(str(out_path))

    if args.export_json is not None:
        # Use exact filename provided by user, no suffix
        out_path = Path(args.export_json)
        with out_path.open("w", encoding="utf-8") as fobj:
            json.dump(all_summaries, fobj, indent=2)

    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
