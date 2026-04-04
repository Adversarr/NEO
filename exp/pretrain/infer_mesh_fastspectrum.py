from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyfastspectrum
import scipy.sparse as sp
import trimesh

from g2pt.data.transforms import normalize_pc
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_ground_truth, solve_gev_subspace
from g2pt.utils.mesh_feats import mesh_laplacian
from g2pt.utils.ortho_operations import qr_orthogonalization


SAMPLE_TYPES = {
    "poisson-disk": pyfastspectrum.SamplingType.Sample_Poisson_Disk,
    "farthest-point": pyfastspectrum.SamplingType.Sample_Farthest_Point,
    "random": pyfastspectrum.SamplingType.Sample_Random,
}

BACKENDS = {
    "auto": pyfastspectrum.SolverBackend.Auto,
    "cpu": pyfastspectrum.SolverBackend.CPU,
    "cuda": pyfastspectrum.SolverBackend.CUDA,
}


def subspace_loss(U: np.ndarray | sp.spmatrix, V: np.ndarray, m_diag: np.ndarray) -> float:
    mv = m_diag[..., None] * V
    term = U.T @ mv
    norm_sq = np.sum(np.asarray(term, dtype=np.float64) ** 2)
    k = V.shape[1]
    return float(1.0 - (1.0 / k) * norm_sq)


def _best_match(pred: np.ndarray, gt: np.ndarray, mass: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"pred and gt must be 2D, got {pred.shape}, {gt.shape}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"N mismatch: pred {pred.shape} vs gt {gt.shape}")

    sim = np.abs(outer_cosine_similarity(pred, gt, M=mass))
    best_idx = np.argmax(sim, axis=0).astype(np.int64)
    best_score = sim[best_idx, np.arange(sim.shape[1])].astype(np.float32)

    if mass is None:
        dots = np.sum(pred[:, best_idx] * gt, axis=0)
    else:
        mass_vec = np.asarray(mass, dtype=np.float64).reshape(-1)
        dots = np.sum((mass_vec[:, None] * pred[:, best_idx]) * gt, axis=0)
    sign = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
    return best_idx, best_score, sign


def _load_any_mesh(path: Path) -> tuple[trimesh.Trimesh | trimesh.PointCloud, np.ndarray, np.ndarray | None]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [geom for geom in loaded.geometry.values()]
        if not geoms:
            raise ValueError(f"No geometry found in {path}")

        meshes: list[trimesh.Trimesh] = []
        point_clouds: list[trimesh.PointCloud] = []
        for geom in geoms:
            if isinstance(geom, trimesh.Trimesh):
                if geom.vertices.size > 0 and geom.faces.size > 0:
                    meshes.append(geom)
            elif isinstance(geom, trimesh.PointCloud):
                if geom.vertices.size > 0:
                    point_clouds.append(geom)

        if meshes:
            merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            vertices = np.asarray(merged.vertices, dtype=np.float32)
            faces = np.asarray(merged.faces, dtype=np.int64)
            return merged, vertices, faces
        if point_clouds:
            merged_pc = trimesh.points.PointCloud(
                vertices=np.concatenate([np.asarray(pc.vertices) for pc in point_clouds], axis=0)
            )
            vertices = np.asarray(merged_pc.vertices, dtype=np.float32)
            return merged_pc, vertices, None
        raise ValueError(f"No supported geometry found in {path}")

    if isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int64) if loaded.faces is not None and len(loaded.faces) > 0 else None
        return loaded, vertices, faces

    if isinstance(loaded, trimesh.PointCloud):
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        return loaded, vertices, None

    raise ValueError(f"Unsupported trimesh type: {type(loaded)} for {path}")


def _safe_case_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _export_original_ply(
    loaded: trimesh.Trimesh | trimesh.PointCloud,
    vertices_raw: np.ndarray,
    faces_raw: np.ndarray | None,
    output_path: Path,
) -> None:
    try:
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
        max_norm = np.max(np.linalg.norm(vertices, axis=1, keepdims=True))
        vertices = vertices / max_norm
        trimesh.Trimesh(vertices=vertices, faces=faces_raw, process=False).export(str(output_path))
    except Exception:
        if faces_raw is not None and faces_raw.size > 0:
            trimesh.Trimesh(vertices=vertices_raw, faces=faces_raw, process=False).export(str(output_path))
        else:
            trimesh.PointCloud(vertices=vertices_raw).export(str(output_path))


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run FastSpectrum on meshes and save per-case baseline results.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing the mesh files.")
    parser.add_argument("--input_dir", type=str, default=None, help="Alias of --data_dir.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Alias of --out_dir.")
    parser.add_argument("--glob", type=str, default="*", help="Glob pattern to match mesh files.")
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of modes used for pairwise comparison. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of FastSpectrum samples / reduced basis size.",
    )
    parser.add_argument(
        "--sample-type",
        choices=tuple(SAMPLE_TYPES),
        default="farthest-point",
        help="Sampling strategy for FastSpectrum.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(BACKENDS),
        default="cpu",
        help="FastSpectrum solver backend.",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    data_dir = args.data_dir or args.input_dir
    out_dir = args.out_dir or args.output_dir
    missing: list[str] = []
    if not data_dir:
        missing.append("--data_dir/--input_dir")
    if not out_dir:
        missing.append("--out_dir/--output_dir")
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    arg_k = args.k if args.k > 0 else args.num_samples
    if arg_k > args.num_samples:
        parser.error(f"--k ({arg_k}) cannot exceed --num-samples ({args.num_samples})")

    data_dir_path = Path(data_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    candidates = sorted(data_dir_path.glob(args.glob))
    mesh_files = [path for path in candidates if path.is_file() and path.suffix.lower() in [".ply", ".obj", ".off"]]
    print(f"Found {len(mesh_files)} meshes in {data_dir_path}")

    fastspectrum = pyfastspectrum.FastSpectrum()

    for index, mesh_path in enumerate(mesh_files):
        case_name = _safe_case_name(mesh_path)
        print(f"Processing {index + 1}/{len(mesh_files)}: {case_name}")
        case_dir = out_dir_path / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        input_dir = case_dir / "input"
        input_dir.mkdir(exist_ok=True)
        inferred_dir = case_dir / "inferred"
        inferred_dir.mkdir(exist_ok=True)

        try:
            loaded, vertices_raw, faces_raw = _load_any_mesh(mesh_path)
        except Exception as exc:
            print(f"Warning: failed to load {mesh_path}: {exc}")
            continue

        _export_original_ply(loaded, vertices_raw, faces_raw, case_dir / "original.ply")

        vertices = normalize_pc(vertices_raw, 0).astype(np.float64)
        faces = faces_raw

        np.save(input_dir / "points.npy", vertices.astype(np.float32))
        if faces is not None:
            np.save(input_dir / "faces.npy", faces.astype(np.int64))

        if faces is None or faces.size == 0:
            print(f"Warning: {mesh_path} has no faces, skip FastSpectrum mesh inference.")
            continue

        lap_start = perf_counter()
        L, M = mesh_laplacian(vertices, faces.astype(np.int64))
        lap_time = perf_counter() - lap_start

        mass = sp.csr_matrix(M).diagonal().astype(np.float32)
        L = sp.csr_matrix(L).astype(np.float64)
        M = sp.csr_matrix(M).astype(np.float64)

        mesh_gev_start = perf_counter()
        mesh_eval, mesh_evec = solve_gev_ground_truth(L=L, M=M, k=arg_k)
        mesh_gev_time = perf_counter() - mesh_gev_start
        mesh_eval = np.asarray(mesh_eval, dtype=np.float64)
        mesh_evec = np.asarray(mesh_evec, dtype=np.float64)

        np.save(input_dir / "mass.npy", mass.astype(np.float32))
        np.save(inferred_dir / "mesh_eval.npy", mesh_eval.astype(np.float32))
        np.save(inferred_dir / "mesh_evec.npy", mesh_evec.astype(np.float32))

        try:
            fast_start = perf_counter()
            basis, reduced_eigvecs, reduced_eigvals = fastspectrum.compute_eigenpairs(
                vertices,
                faces.astype(np.int32),
                args.num_samples,
                sample_type=SAMPLE_TYPES[args.sample_type],
                backend=BACKENDS[args.backend],
            )
            fast_time = perf_counter() - fast_start
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            continue

        fast_subspace = np.asarray(basis @ reduced_eigvecs, dtype=np.float64)
        fast_subspace = qr_orthogonalization(fast_subspace, mass.reshape(-1, 1)).astype(np.float64)
        fast_eval, fast_evec = solve_gev_subspace(L, M, fast_subspace)
        fast_order = np.argsort(fast_eval)
        fast_eval = fast_eval[fast_order][:arg_k]
        fast_evec = fast_evec[:, fast_order][:, :arg_k]
        loss_val = subspace_loss(fast_evec, mesh_evec, mass)

        match_fast_to_mesh, score_mesh_fast, sign_fast_to_mesh = _best_match(fast_evec, mesh_evec, mass)
        fast_eval_aligned = fast_eval[match_fast_to_mesh]
        fast_evec_aligned = fast_evec[:, match_fast_to_mesh] * sign_fast_to_mesh.reshape(1, -1)

        eval_relerr = np.abs((fast_eval_aligned - mesh_eval) / (mesh_eval + 1e-8))
        if eval_relerr.size > 0:
            eval_relerr[0] = 0.0

        residual_fast = np.linalg.norm(
            L @ fast_evec_aligned - M @ fast_evec_aligned * fast_eval_aligned.reshape(1, -1),
            axis=0,
        )
        avg_residual = float(np.mean(residual_fast)) if residual_fast.size > 0 else 0.0

        low_prec_start = perf_counter()
        solve_gev_ground_truth(L, M, k=arg_k, tol=avg_residual)
        same_residual_time = perf_counter() - low_prec_start

        np.save(inferred_dir / "fastspectrum_eval.npy", fast_eval_aligned.astype(np.float32))
        np.save(inferred_dir / "fastspectrum_evec.npy", fast_evec_aligned.astype(np.float32))
        np.save(inferred_dir / "match_fastspectrum_to_mesh.npy", match_fast_to_mesh)
        np.save(inferred_dir / "sign_fastspectrum_to_mesh.npy", sign_fast_to_mesh)
        np.save(inferred_dir / "score_mesh_fastspectrum.npy", score_mesh_fast)

        print(f"[{index + 1}/{len(mesh_files)}] {mesh_path.name} FastSpectrum time: {fast_time:.4f} seconds")
        print(f"[{index + 1}/{len(mesh_files)}] {mesh_path.name} Mesh Laplacian time: {lap_time:.4f} seconds")
        print(f"[{index + 1}/{len(mesh_files)}] {mesh_path.name} Mesh GEV time: {mesh_gev_time:.4f} seconds")
        print(
            f"[{index + 1}/{len(mesh_files)}] {mesh_path.name} Scores mesh↔fast: "
            f"mean={float(score_mesh_fast.mean()):.4f}, median={float(np.median(score_mesh_fast)):.4f}, loss={float(loss_val):.4f}"
        )

        metrics = {
            "times": {
                "mesh_laplacian": float(lap_time),
                "mesh_gev": float(mesh_gev_time),
                "fastspectrum": float(fast_time),
                "same_residual_gev": float(same_residual_time),
            },
            "scores": {
                "mesh_vs_fastspectrum": score_mesh_fast.tolist(),
                "subspace_loss": float(loss_val),
                "eval_relerr": eval_relerr.tolist(),
                "residual": residual_fast.tolist(),
            },
        }
        information_json = {
            "file_name": mesh_path.absolute().as_posix(),
            "n_points": int(vertices.shape[0]),
            "n_verts": int(vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
            "arg_k": int(arg_k),
            "fastspectrum_k": int(args.num_samples),
            "times": metrics["times"],
            "scores": metrics["scores"],
            "fastspectrum": {
                "config": {
                    "num_samples": int(args.num_samples),
                    "sample_type": args.sample_type,
                    "backend": args.backend,
                },
                **metrics,
            },
        }
        with (inferred_dir / "fastspectrum_results.json").open("w", encoding="utf-8") as handle:
            json.dump(information_json, handle, indent=4)


if __name__ == "__main__":
    main()
