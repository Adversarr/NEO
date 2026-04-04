from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import scipy.sparse as sp

from pyamgx_eigen import solve_smallest_generalized_eigenpairs

from g2pt.data.transforms import normalize_pc
from g2pt.utils.gev import outer_cosine_similarity
from robust_laplacian import point_cloud_laplacian


def _best_match(
    pred: np.ndarray,
    gt: np.ndarray,
    mass: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    sim = np.abs(outer_cosine_similarity(pred, gt, M=mass))
    best_idx = np.argmax(sim, axis=0).astype(np.int64)
    best_score = sim[best_idx, np.arange(sim.shape[1])].astype(np.float32)

    if mass is None:
        dots = np.sum(pred[:, best_idx] * gt, axis=0)
    else:
        m = np.asarray(mass, dtype=np.float64).reshape(-1)
        dots = np.sum((m[:, None] * pred[:, best_idx]) * gt, axis=0)
    sign = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
    return best_idx, best_score, sign


def main() -> None:
    parser = ArgumentParser(description="Quick pyamgx experiments on point-cloud generalized eigenproblems.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=1.0e-6)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--subspace_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    folders = sorted(data_dir.glob(args.glob))
    subspace_size = args.subspace_size if args.subspace_size > 0 else None

    print(f"Found {len(folders)} samples in {data_dir}")
    print(
        f"pyamgx params: k={args.k}, sigma={args.sigma}, tol={args.tol}, "
        f"max_iters={args.max_iters}, subspace_size={subspace_size}, seed={args.seed}"
    )

    for i, case in enumerate(folders):
        if not case.is_dir():
            continue

        pc_file = case / "sample_points.npy"
        if not pc_file.exists():
            print(f"[skip] {case}: missing sample_points.npy")
            continue

        result_path = case / "results_pyamgx_exp.json"
        if result_path.exists() and not args.force:
            print(f"[skip] {case}: results_pyamgx_exp.json already exists")
            continue

        mesh_evec_path = case / "input" / "mesh_evec.npy"
        if not mesh_evec_path.exists():
            mesh_evec_path = case / "temp" / "mesh_evec.npy"
        if not mesh_evec_path.exists():
            mesh_evec_path = case / "mesh_evec.npy"

        pc_raw = np.load(pc_file)
        pc = normalize_pc(pc_raw, 0)

        lap_start = perf_counter()
        L, M = point_cloud_laplacian(pc)
        lap_time = perf_counter() - lap_start

        L = sp.csr_matrix(L, dtype=np.float64)
        M_csr = sp.csr_matrix(M, dtype=np.float64)
        mass = M.diagonal().astype(np.float64)

        solve_start = perf_counter()
        evals, evecs, info = solve_smallest_generalized_eigenpairs(
            L + args.sigma * M_csr,
            M_csr,
            k=args.k,
            tol=args.tol,
            max_iters=args.max_iters,
            subspace_size=subspace_size,
            seed=args.seed,
        )
        solve_time = perf_counter() - solve_start

        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        residual = np.linalg.norm(L @ evecs - M_csr @ evecs * evals.reshape(1, -1), axis=0)
        mesh_scores = None
        matched_evals = None
        matched_residual = None

        print(f"\n== {case} ==")
        if i == 0:
            print(f"first sample shape: {pc.shape}, dtype={pc.dtype}")
        print(f"laplacian time: {lap_time:.4f}s")
        print(f"pyamgx iterations: {info['iterations']}")
        print(f"converged: {info['converged']}")
        print(f"solve time: {solve_time:.4f}s")
        print(f"eval head: {evals[: min(5, len(evals))]}")
        print(
            "residual: "
            f"mean={float(residual.mean()):.4e}, median={float(np.median(residual)):.4e}, "
            f"max={float(residual.max()):.4e}"
        )
        print(f"orthogonality_error: {info['orthogonality_error']:.4e}")

        if mesh_evec_path.exists():
            mesh_evec_full = np.load(mesh_evec_path)
            if (
                mesh_evec_full.ndim == 2
                and mesh_evec_full.shape[0] == pc.shape[0]
                and mesh_evec_full.shape[1] >= args.k
            ):
                mesh_evec = np.asarray(mesh_evec_full[:, : args.k], dtype=np.float64)
                match_pc_to_mesh, score_mesh_pc, sign_pc_to_mesh = _best_match(evecs, mesh_evec, mass)
                aligned_evecs = evecs[:, match_pc_to_mesh] * sign_pc_to_mesh.reshape(1, -1)
                matched_evals = evals[match_pc_to_mesh]
                aligned_residual = np.linalg.norm(
                    L @ aligned_evecs - M_csr @ aligned_evecs * matched_evals.reshape(1, -1),
                    axis=0,
                )
                mesh_scores = score_mesh_pc
                matched_residual = aligned_residual
                print(
                    "mesh<->pyamgx cosine: "
                    f"mean={float(score_mesh_pc.mean()):.4f}, "
                    f"median={float(np.median(score_mesh_pc)):.4f}, "
                    f"min={float(score_mesh_pc.min()):.4f}"
                )
                print(f"matched eval head: {matched_evals[: min(5, len(match_pc_to_mesh))]}")
                print(
                    "matched residual: "
                    f"mean={float(aligned_residual.mean()):.4e}, "
                    f"max={float(aligned_residual.max()):.4e}"
                )
            else:
                print(f"mesh_evec shape not usable: {mesh_evec_full.shape}")
        else:
            print("mesh_evec.npy not found; skipping mesh comparison")

        scores = {
            "mesh_vs_pointcloud": mesh_scores.tolist() if mesh_scores is not None else [],
            "mesh_vs_network": [],
            "pointcloud_vs_network": [],
            "subspace_loss": None,
            "eval_relerr": [],
        }
        precisions = {
            "pyamgx": {
                "loss": None,
                "times": {
                    "forward": 0.0,
                    "qr": 0.0,
                    "network_gev": float(solve_time),
                    "pointcloud_gev": float(solve_time),
                    "same_residual_gev": 0.0,
                },
                "scores": {
                    "mesh_vs_pointcloud": scores["mesh_vs_pointcloud"],
                    "mesh_vs_network": [],
                    "pointcloud_vs_network": [],
                    "eval_relerr": [],
                },
                "solver": {
                    "iterations": int(info["iterations"]),
                    "converged": bool(info["converged"]),
                    "sigma": float(args.sigma),
                    "tol": float(args.tol),
                    "max_iters": int(args.max_iters),
                    "subspace_size": subspace_size,
                    "seed": int(args.seed),
                    "orthogonality_error": float(info["orthogonality_error"]),
                },
                "residual": {
                    "all": residual.tolist(),
                    "matched": matched_residual.tolist() if matched_residual is not None else [],
                },
                "eigenvalues": {
                    "all": evals.tolist(),
                    "matched": matched_evals.tolist() if matched_evals is not None else [],
                },
            }
        }
        information_json = {
            "file_name": pc_file.absolute().as_posix(),
            "n_points": int(pc.shape[0]),
            "arg_k": int(args.k),
            "net_k": int(args.k),
            "times": {
                "forward": 0.0,
                "qr": 0.0,
                "network_gev": float(solve_time),
                "pointcloud_gev": float(solve_time),
                "same_residual_gev": 0.0,
            },
            "solver": {
                "iterations": int(info["iterations"]),
                "converged": bool(info["converged"]),
            },
            "scores": scores,
            "precisions": precisions,
        }
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(information_json, f, indent=4)


if __name__ == "__main__":
    main()
