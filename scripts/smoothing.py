"""Spectral denoising using given basis"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import trimesh

try:
    from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian
except Exception:
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    try:
        from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian
    except Exception:
        mesh_laplacian = None
        point_cloud_laplacian = None


FilterMode = Literal["truncate", "heat"]
HeatEvalsMode = Literal["provided", "subspace"]


@dataclass(frozen=True)
class BasisData:
    evec: np.ndarray
    evals: np.ndarray | None
    name: str
    evec_file: str
    eval_file: str | None


def _as_2d_evec(evec: np.ndarray) -> np.ndarray:
    evec = np.asarray(evec)
    if evec.ndim == 3 and evec.shape[0] == 1:
        evec = evec[0]
    if evec.ndim != 2:
        raise ValueError(f"basis must be 2D (N,k), got shape={evec.shape}")
    return np.asarray(evec, dtype=np.float64)


def _load_mesh_or_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npy":
        pts = np.load(path).astype(np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points.npy to have shape (N,3), got {pts.shape}")
        return pts, None

    if path.suffix.lower() in {".obj", ".ply", ".off", ".stl", ".glb", ".gltf"}:
        m = trimesh.load(path, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        if not hasattr(m, "vertices"):
            raise ValueError(f"Failed to load mesh/points from {path}")
        vertices = np.asarray(m.vertices, dtype=np.float64)
        faces = None
        if hasattr(m, "faces") and m.faces is not None and len(m.faces) > 0:
            faces = np.asarray(m.faces, dtype=np.int64)
        return vertices, faces

    raise ValueError(f"Unsupported input file: {path}")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_points_faces_mass_from_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    input_dir = case_dir / "input"
    points = None
    faces = None
    mass = None

    if (input_dir / "points.npy").exists():
        points = np.load(input_dir / "points.npy").astype(np.float64)
    elif (case_dir / "sample_points.npy").exists():
        points = np.load(case_dir / "sample_points.npy").astype(np.float64)
    else:
        raise FileNotFoundError(f"Could not find input/points.npy or sample_points.npy under {case_dir}")

    if (input_dir / "faces.npy").exists():
        faces = np.load(input_dir / "faces.npy").astype(np.int64)

    if (input_dir / "mass.npy").exists():
        mass = np.load(input_dir / "mass.npy").astype(np.float64).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape} from {case_dir}")
    if mass is not None and mass.shape[0] != points.shape[0]:
        raise ValueError(f"mass length mismatch: {mass.shape} vs N={points.shape[0]}")

    return points, faces, mass


def _load_basis_from_case(case_dir: Path, basis: str) -> BasisData:
    input_dir = case_dir / "input"
    inferred_dir = case_dir / "inferred"

    def _load_pair(evec_path: Path, eval_path: Path | None, name: str) -> BasisData:
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

    if basis == "net_fp16":
        evec_path = inferred_dir / "net_evec_fp16.npy"
        eval_path = inferred_dir / "net_eval_fp16.npy"
        if not evec_path.exists():
            raise FileNotFoundError(f"net_evec_fp16.npy not found under {inferred_dir}")
        return _load_pair(evec_path, eval_path if eval_path.exists() else None, "net_fp16")

    if basis in {"net_pred", "net_pred_original", "net_pred_original_fp32"}:
        evec_path = inferred_dir / "net_pred_original_fp32.npy"
        if not evec_path.exists():
            evec_path = inferred_dir / "net_pred_original.npy"
        if not evec_path.exists():
            raise FileNotFoundError(f"net_pred_original*.npy not found under {inferred_dir}")
        return _load_pair(evec_path, None, "net_pred_original_fp32")

    if basis in {"net_pred_original_fp16"}:
        evec_path = inferred_dir / "net_pred_original_fp16.npy"
        if not evec_path.exists():
            raise FileNotFoundError(f"net_pred_original_fp16.npy not found under {inferred_dir}")
        return _load_pair(evec_path, None, "net_pred_original_fp16")

    raise ValueError(f"Unknown basis={basis!r}")


def _estimate_mass(points: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    n = int(points.shape[0])
    if faces is not None and faces.size > 0 and mesh_laplacian is not None:
        _, M = mesh_laplacian(points, faces)
        mass = np.asarray(M.diagonal(), dtype=np.float64).reshape(-1)
        if mass.shape[0] == n:
            return mass
    if point_cloud_laplacian is not None:
        _, M = point_cloud_laplacian(points)
        mass = np.asarray(M.diagonal(), dtype=np.float64).reshape(-1)
        if mass.shape[0] == n:
            return mass
    return np.ones(n, dtype=np.float64)


def _spectral_project(points: np.ndarray, evec: np.ndarray, mass: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    E = np.asarray(evec, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {p.shape}")
    if E.ndim != 2:
        raise ValueError(f"evec must be (N,k), got {E.shape}")
    if E.shape[0] != p.shape[0]:
        raise ValueError(f"N mismatch: points {p.shape} vs evec {E.shape}")
    if m.shape[0] != p.shape[0]:
        raise ValueError(f"mass length mismatch: {m.shape} vs N={p.shape[0]}")

    rhs = E.T @ (p * m[:, None])
    gram = E.T @ (E * m[:, None])
    k = int(E.shape[1])
    gram = gram + float(ridge) * np.eye(k, dtype=np.float64)
    coeff = np.linalg.solve(gram, rhs)
    return coeff


def _spectral_reconstruct(evec: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    E = np.asarray(evec, dtype=np.float64)
    c = np.asarray(coeff, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"evec must be (N,k), got {E.shape}")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coeff must be (k,3), got {c.shape}")
    if E.shape[1] != c.shape[0]:
        raise ValueError(f"k mismatch: evec {E.shape} vs coeff {c.shape}")
    return E @ c


def _apply_filter(coeff: np.ndarray, mode: FilterMode, evals: np.ndarray | None, t: float | None) -> np.ndarray:
    c = np.asarray(coeff, dtype=np.float64)
    if mode == "truncate":
        return c
    if mode == "heat":
        if evals is None:
            raise ValueError("Heat filter requires eigenvalues, but evals is missing for the chosen basis.")
        if t is None:
            raise ValueError("Heat filter requires --t.")
        lam = np.asarray(evals, dtype=np.float64).reshape(-1)
        if lam.shape[0] != c.shape[0]:
            raise ValueError(f"evals length mismatch: {lam.shape} vs k={c.shape[0]}")
        f = np.exp(-float(t) * lam)[:, None]
        return c * f
    raise ValueError(f"Unknown filter mode: {mode}")


def _build_laplacian(points: np.ndarray, faces: np.ndarray | None):
    pts = np.asarray(points, dtype=np.float64)
    if faces is not None and faces.size > 0:
        if mesh_laplacian is None:
            raise RuntimeError("mesh_laplacian is unavailable; cannot compute subspace eigenvalues.")
        return mesh_laplacian(pts, np.asarray(faces, dtype=np.int64))
    if point_cloud_laplacian is None:
        raise RuntimeError("point_cloud_laplacian is unavailable; cannot compute subspace eigenvalues.")
    return point_cloud_laplacian(pts)


def _subspace_eigendecomp(
    evec: np.ndarray,
    L,
    M,
    ridge: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(evec, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"evec must be (N,k), got {E.shape}")
    try:
        from scipy import linalg as sla  # type: ignore
    except Exception:
        sla = None

    Lr = E.T @ (L @ E)
    Mr = E.T @ (M @ E)
    Lr = 0.5 * (Lr + Lr.T)
    Mr = 0.5 * (Mr + Mr.T)
    Mr = Mr + float(ridge) * np.eye(Mr.shape[0], dtype=np.float64)

    if sla is not None:
        evals, U = sla.eigh(Lr, Mr)
    else:
        A = np.linalg.solve(Mr, Lr)
        evals, U = np.linalg.eig(A)
        evals = np.real(evals)
        U = np.real(U)

    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64).reshape(-1)
    U = np.asarray(U[:, order], dtype=np.float64)
    evals = np.maximum(evals, 0.0)
    return evals, U


def _preserve_center_and_scale(ref: np.ndarray, target: np.ndarray, mode: Literal["none", "rms"] = "rms") -> np.ndarray:
    if mode == "none":
        return target
    if mode != "rms":
        raise ValueError(f"Unknown preserve mode: {mode}")

    ref = np.asarray(ref, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    ref_c = ref.mean(axis=0, keepdims=True)
    tgt_c = tgt.mean(axis=0, keepdims=True)
    ref0 = ref - ref_c
    tgt0 = tgt - tgt_c
    ref_s = float(np.sqrt(np.mean(np.sum(ref0 * ref0, axis=1))))
    tgt_s = float(np.sqrt(np.mean(np.sum(tgt0 * tgt0, axis=1))))
    if ref_s <= 1e-12 or tgt_s <= 1e-12:
        return tgt - tgt_c + ref_c
    return (tgt0 * (ref_s / tgt_s)) + ref_c


def _export_geometry(points: np.ndarray, faces: np.ndarray | None, out_path: Path) -> None:
    out_path = Path(out_path)
    if faces is not None and faces.size > 0:
        trimesh.Trimesh(vertices=points, faces=faces, process=False).export(out_path.as_posix())
    else:
        trimesh.PointCloud(vertices=points).export(out_path.as_posix())


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Spectral denoising / fairing using a provided spectral basis.")
    ap.add_argument("--infer_case_dir", type=str, default=None, help="Inference case directory containing input/ and inferred/.")
    ap.add_argument("--points_path", type=str, default=None, help="Optional override points source (mesh/ply/obj or npy).")
    ap.add_argument("--basis", type=str, default="net_fp32", help="Basis key: mesh, pc, net_fp32, net_fp16, net_pred_original_fp32.")
    ap.add_argument("--basis_path", type=str, default=None, help="Explicit basis .npy path (overrides --basis).")
    ap.add_argument("--evals_path", type=str, default=None, help="Explicit eigenvalues .npy path (optional).")
    ap.add_argument("--mass_path", type=str, default=None, help="Explicit mass .npy path (optional).")
    ap.add_argument("--basis_k", type=int, default=0, help="Truncate basis to first k modes (0=all).")

    ap.add_argument("--filter", type=str, default="heat", choices=["heat", "truncate"], help="Spectral filtering mode.")
    ap.add_argument("--t", type=float, default=None, help="Heat kernel time parameter when --filter=heat.")
    ap.add_argument("--ridge", type=float, default=1e-10, help="Ridge for Gram matrix inversion in projection.")
    ap.add_argument("--heat_evals", type=str, default="subspace", choices=["subspace", "provided"], help="Eigenvalues source for heat filter.")

    ap.add_argument("--add_noise_sigma", type=float, default=0.0, help="Add i.i.d. Gaussian noise to input points.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for noise.")

    ap.add_argument("--preserve", type=str, default="rms", choices=["none", "rms"], help="Preserve center/scale after filtering.")
    ap.add_argument("--out_dir", type=str, default="output_smoothing", help="Output directory.")
    ap.add_argument("--export_ply", action="store_true", help="Export PLY of noisy and smoothed geometry.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    case_dir = Path(args.infer_case_dir) if args.infer_case_dir is not None else None
    if case_dir is None and args.points_path is None:
        raise ValueError("Provide at least one of --infer_case_dir or --points_path.")

    points_ref = None
    faces = None
    mass = None
    if case_dir is not None:
        points_ref, faces, mass = _load_points_faces_mass_from_case(case_dir)

    if args.points_path is not None:
        points, faces_override = _load_mesh_or_points(Path(args.points_path))
        if points_ref is not None and points.shape[0] != points_ref.shape[0]:
            raise ValueError(f"points_path N mismatch: {points.shape[0]} vs case points {points_ref.shape[0]}")
        if faces_override is not None:
            faces = faces_override
        points_ref = points

    if points_ref is None:
        raise ValueError("Failed to load points.")

    if args.mass_path is not None:
        mass = np.load(args.mass_path).astype(np.float64).reshape(-1)
    if mass is None:
        mass = _estimate_mass(points_ref, faces)
    if mass.shape[0] != points_ref.shape[0]:
        raise ValueError(f"mass length mismatch: {mass.shape} vs N={points_ref.shape[0]}")

    if args.basis_path is not None:
        evec = _as_2d_evec(np.load(args.basis_path))
        evals = None
        if args.evals_path is not None:
            evals = np.load(args.evals_path).astype(np.float64).reshape(-1)
        basis_data = BasisData(
            evec=evec,
            evals=evals,
            name="custom",
            evec_file=str(Path(args.basis_path).as_posix()),
            eval_file=str(Path(args.evals_path).as_posix()) if args.evals_path is not None else None,
        )
    else:
        if case_dir is None:
            raise ValueError("Using --basis requires --infer_case_dir unless --basis_path is provided.")
        basis_data = _load_basis_from_case(case_dir, args.basis)

    evec = basis_data.evec
    evals = basis_data.evals
    if args.basis_k and args.basis_k > 0:
        k = int(args.basis_k)
        if k > evec.shape[1]:
            raise ValueError(f"basis_k={k} exceeds basis width={evec.shape[1]}")
        evec = evec[:, :k]
        if evals is not None:
            evals = evals[:k]

    if evec.shape[0] != points_ref.shape[0]:
        raise ValueError(f"basis N mismatch: evec {evec.shape} vs points {points_ref.shape}")

    rng = np.random.default_rng(int(args.seed))
    points_noisy = np.asarray(points_ref, dtype=np.float64)
    if float(args.add_noise_sigma) > 0.0:
        points_noisy = points_noisy + rng.normal(scale=float(args.add_noise_sigma), size=points_noisy.shape)

    t0 = perf_counter()
    coeff = _spectral_project(points_noisy, evec, mass, ridge=float(args.ridge))
    if args.filter == "heat":
        if args.t is None:
            raise ValueError("Heat filter requires --t.")
        if args.heat_evals == "provided":
            coeff_f = _apply_filter(coeff, mode="heat", evals=evals, t=args.t)
        else:
            L, M = _build_laplacian(points_ref, faces)
            sub_evals, U = _subspace_eigendecomp(evec, L=L, M=M)
            c_rot = np.linalg.solve(U, coeff)
            f = np.exp(-float(args.t) * sub_evals)[:, None]
            c_rot_f = c_rot * f
            coeff_f = U @ c_rot_f
    else:
        coeff_f = _apply_filter(coeff, mode="truncate", evals=evals, t=args.t)
    points_smooth = _spectral_reconstruct(evec, coeff_f)
    points_smooth = _preserve_center_and_scale(points_noisy, points_smooth, mode=args.preserve)
    t1 = perf_counter()

    np.save(out_dir / "points_noisy.npy", points_noisy.astype(np.float32))
    np.save(out_dir / "points_smooth.npy", points_smooth.astype(np.float32))
    if faces is not None:
        np.save(out_dir / "faces.npy", np.asarray(faces, dtype=np.int64))
    np.save(out_dir / "mass.npy", np.asarray(mass, dtype=np.float32))

    metrics = {}
    if float(args.add_noise_sigma) > 0.0:
        metrics["rmse_noisy_to_ref"] = _rmse(points_noisy, points_ref)
        metrics["rmse_smooth_to_ref"] = _rmse(points_smooth, points_ref)

    results = {
        "case_dir": str(case_dir.as_posix()) if case_dir is not None else None,
        "points_path": args.points_path,
        "basis": {
            "name": basis_data.name,
            "evec_file": basis_data.evec_file,
            "eval_file": basis_data.eval_file,
            "k": int(evec.shape[1]),
        },
        "filter": {"mode": args.filter, "t": args.t, "heat_evals": args.heat_evals if args.filter == "heat" else None},
        "noise": {"sigma": float(args.add_noise_sigma), "seed": int(args.seed)},
        "preserve": args.preserve,
        "time_total_sec": float(t1 - t0),
        "metrics": metrics,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.export_ply:
        _export_geometry(points_noisy, faces, out_dir / "noisy.ply")
        _export_geometry(points_smooth, faces, out_dir / "smooth.ply")

    print(f"Saved outputs to {out_dir}")
    if metrics:
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6g}")


if __name__ == "__main__":
    main()
