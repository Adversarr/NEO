from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pyspec"))

from pyspec import (
    configured_num_threads,
    get_num_threads,
    openmp_enabled,
    set_num_threads,
    solve_lowest_generalized_eigenpairs,
)

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
    parser = ArgumentParser(description="Quick pyspec experiments on point-cloud generalized eigenproblems.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=1.0e-8)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--ncv", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    folders = sorted(data_dir.glob(args.glob))
    ncv = args.ncv if args.ncv > 0 else None
    if args.num_threads > 0:
        set_num_threads(args.num_threads)
    active_threads = get_num_threads()

    print(f"Found {len(folders)} samples in {data_dir}")
    print(
        f"pyspec params: k={args.k}, sigma={args.sigma}, tol={args.tol}, "
        f"maxit={args.maxit}, ncv={ncv}"
    )
    print(
        "threading: "
        f"openmp_enabled={openmp_enabled}, "
        f"configured_num_threads={configured_num_threads}, "
        f"active_num_threads={active_threads}"
    )

    for i, case in enumerate(folders):
        if not case.is_dir():
            continue

        pc_file = case / "sample_points.npy"
        if not pc_file.exists():
            print(f"[skip] {case}: missing sample_points.npy")
            continue

        result_path = case / "results_spectra_exp.json"
        if result_path.exists() and not args.force:
            print(f"[skip] {case}: results_spectra_exp.json already exists")
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

        # Row-major CSR storage lets Eigen use its OpenMP sparse matvec path.
        L = sp.csr_matrix(L, dtype=np.float64)
        M = sp.csr_matrix(M, dtype=np.float64)
        mass = M.diagonal().astype(np.float64)

        solve_start = perf_counter()
        result = solve_lowest_generalized_eigenpairs(
            L + args.sigma * M,
            M,
            args.k,
            ncv=ncv,
            sigma=args.sigma,
            maxit=args.maxit,
            tol=args.tol,
        )
        solve_time = perf_counter() - solve_start

        evals = np.asarray(result.eigenvalues, dtype=np.float64)
        evecs = np.asarray(result.eigenvectors, dtype=np.float64)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        residual = np.linalg.norm(L @ evecs - M @ evecs * evals.reshape(1, -1), axis=0)
        mesh_scores = None
        matched_evals = None
        matched_residual = None

        print(f"\n== {case} ==")
        if i == 0:
            print(f"first sample shape: {pc.shape}, dtype={pc.dtype}")
        print(f"laplacian time: {lap_time:.4f}s")
        print(f"pyspec status: {result.status}")
        print(f"converged: {result.nconv}/{args.k}")
        print(f"solve time: {solve_time:.4f}s")
        print(f"eval head: {evals[: min(5, len(evals))]}")
        print(
            "residual: "
            f"mean={float(residual.mean()):.4e}, median={float(np.median(residual)):.4e}, "
            f"max={float(residual.max()):.4e}"
        )

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
                    L @ aligned_evecs - M @ aligned_evecs * matched_evals.reshape(1, -1),
                    axis=0,
                )
                mesh_scores = score_mesh_pc
                matched_residual = aligned_residual
                print(
                    "mesh<->pyspec cosine: "
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
            "pyspec": {
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
                    "status": result.status,
                    "nconv": int(result.nconv),
                    "sigma": float(args.sigma),
                    "tol": float(args.tol),
                    "maxit": int(args.maxit),
                    "ncv": ncv,
                    "num_threads": int(active_threads),
                    "openmp_enabled": bool(openmp_enabled),
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
                "num_threads": int(active_threads),
                "openmp_enabled": bool(openmp_enabled),
            },
            "scores": scores,
            "precisions": precisions,
        }
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(information_json, f, indent=4)


if __name__ == "__main__":
    main()
