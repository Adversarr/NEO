"""
Evaluate checkpoint-driven functional maps on the preprocessed FAUST dataset.

Example:
    uv run python scripts/matcher_faust.py \
        --ckpt checkpoints/Finals/Ours-Small-NoRoPE.ckpt \
        --data-dir ldata/processed_FAUST_corr \
        --out-dir tmp_faust_match
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import h5py
import igl
import numpy as np
import scipy.sparse as sp
import torch
from pyFM.mesh import TriMesh
from pyFM.refine.zoomout import mesh_zoomout_refine_p2p
from pyFM.signatures.HKS_functions import mesh_HKS
from pyFM.signatures.WKS_functions import mesh_WKS
from pyFM.spectral.nn_utils import knn_query
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from g2pt.data.transforms import normalize_pc
from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.gev import solve_gev_from_subspace
from g2pt.utils.mesh_feats import mesh_laplacian


@dataclass
class CaseData:
    name: str
    split: str
    obj_path: str
    vertices: np.ndarray
    faces: np.ndarray
    corres: np.ndarray
    gt_eval: np.ndarray
    gt_evec: np.ndarray
    pred_eval: np.ndarray
    pred_evec: np.ndarray
    mass: np.ndarray
    mesh_gt: TriMesh
    mesh_pred: TriMesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-wide FAUST functional map evaluation with pyFM.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint used for mesh inference.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ldata/processed_FAUST_corr"),
        help="Directory containing preprocess_corr.py HDF5 outputs.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for summary.json and pairs.csv.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for checkpoint inference. Use cuda or cpu.",
    )
    parser.add_argument("--k-process", type=int, default=30, help="Basis dimension used by pyFM matching.")
    parser.add_argument("--signature", type=str, choices=["HKS", "WKS"], default="HKS", help="Descriptor family.")
    parser.add_argument("--wks-num-E", type=int, default=30, help="Descriptor channel count.")
    parser.add_argument("--wks-k", type=int, default=30, help="Basis size used by descriptor construction.")
    parser.add_argument("--zo-k-init", type=int, default=10, help="Initial basis size for ZoomOut.")
    parser.add_argument("--zo-nit", type=int, default=4, help="Number of ZoomOut iterations.")
    parser.add_argument("--zo-step", type=int, default=5, help="ZoomOut basis increment per iteration.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel workers used by pyFM utilities.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging. 0 means all pairs.")
    parser.add_argument(
        "--case-filter",
        type=str,
        default=None,
        help="Optional fnmatch pattern for case names, for example 'tr_reg_06[13]'.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Filename for aggregate metrics JSON under out-dir.",
    )
    parser.add_argument(
        "--pairs-name",
        type=str,
        default="pairs.csv",
        help="Filename for per-pair scalar metrics under out-dir.",
    )
    parser.add_argument(
        "--spectral-name",
        type=str,
        default="spectral_cases.csv",
        help="Filename for per-case spectral comparison metrics under out-dir.",
    )
    return parser.parse_args()


def _load_vlen_array(dataset: h5py.Dataset, index: int, dtype: np.dtype) -> np.ndarray:
    return np.array(dataset[index], dtype=dtype)


def _reshape_vlen(dataset: h5py.Dataset, shape_ds: h5py.Dataset, index: int, dtype: np.dtype) -> np.ndarray:
    flat = _load_vlen_array(dataset, index, dtype=dtype)
    shape = tuple(int(v) for v in shape_ds[index].tolist())
    return flat.reshape(shape)


def _build_vertex_edge_graph(vertices: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    faces = np.asarray(faces, dtype=np.int64)
    undirected = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(undirected, axis=1)
    undirected = np.unique(undirected, axis=0)

    weights = np.linalg.norm(vertices[undirected[:, 0]] - vertices[undirected[:, 1]], axis=1).astype(np.float64)
    src = np.concatenate([undirected[:, 0], undirected[:, 1]])
    dst = np.concatenate([undirected[:, 1], undirected[:, 0]])
    ww = np.concatenate([weights, weights])
    return sp.csr_matrix((ww, (src, dst)), shape=(vertices.shape[0], vertices.shape[0]))


def _compute_geodesic_matrix(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    graph = _build_vertex_edge_graph(vertices, faces)
    area = float(np.sum(igl.doublearea(vertices, faces)) / 2.0)
    norm = float(np.sqrt(max(area, 1e-12)))
    indices = np.arange(vertices.shape[0], dtype=np.int64)
    distances = dijkstra(graph, directed=False, indices=indices)
    return (distances / norm).astype(np.float32)


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _metric_bundle(errors: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.k_process <= 0:
        raise ValueError("--k-process must be positive.")
    required_basis = args.zo_k_init + args.zo_nit * args.zo_step
    if args.k_process < required_basis:
        raise ValueError(f"--k-process={args.k_process} is too small for ZoomOut; need at least {required_basis}.")
    if args.wks_k > args.k_process:
        raise ValueError(f"--wks-k={args.wks_k} cannot exceed --k-process={args.k_process}.")


def _load_model(ckpt_path: Path, device: str) -> PretrainTraining:
    model = PretrainTraining.load_from_checkpoint(ckpt_path.as_posix(), weights_only=False, strict=False)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _infer_rr_eigenpairs(
    model: PretrainTraining,
    vertices: np.ndarray,
    faces: np.ndarray,
    k_process: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = normalize_pc(vertices.astype(np.float32), 0.0)
    L, M = mesh_laplacian(points.astype(np.float64), faces.astype(np.int64))
    mass = M.diagonal().astype(np.float32)
    mesh_tensor = torch.tensor(points, dtype=torch.float32, device=model.device).unsqueeze(0)
    mass_tensor = torch.tensor(mass, dtype=torch.float32, device=model.device).view(1, -1, 1)

    with torch.no_grad():
        subspace, _ = model.forward(mesh_tensor, mass_tensor, return_time=False)

    if subspace.shape[-1] < k_process:
        raise ValueError(
            f"Checkpoint output dimension {subspace.shape[-1]} is smaller than requested k_process={k_process}."
        )

    evals_rr, evecs_rr = solve_gev_from_subspace(
        subspace[0].detach().cpu(),
        torch.from_numpy(points),
        torch.from_numpy(mass),
        k=subspace.shape[-1],
        L=L,
        M=M,
    )
    return evals_rr[:k_process].astype(np.float32), evecs_rr[:, :k_process].astype(np.float32), mass


def _build_matching_mesh(vertices: np.ndarray, faces: np.ndarray, evals: np.ndarray, evecs: np.ndarray) -> TriMesh:
    mesh = TriMesh(vertices.astype(np.float64), faces.astype(np.int64), center=False, area_normalize=False)
    mesh.process(k=int(evecs.shape[1]), intrinsic=False, verbose=False)
    mesh.eigenvalues = np.asarray(evals, dtype=np.float64).reshape(-1)
    mesh.eigenvectors = np.asarray(evecs, dtype=np.float64)
    return mesh


def _spectral_metrics(case: CaseData) -> dict[str, Any]:
    mass = np.asarray(case.mass, dtype=np.float64).reshape(-1)
    pred = np.asarray(case.pred_evec, dtype=np.float64)
    gt = np.asarray(case.gt_evec, dtype=np.float64)
    pred_eval = np.asarray(case.pred_eval, dtype=np.float64)
    gt_eval = np.asarray(case.gt_eval, dtype=np.float64)

    sqrt_mass = np.sqrt(np.maximum(mass, 1e-12)).reshape(-1, 1)
    pred_w = pred * sqrt_mass
    gt_w = gt * sqrt_mass

    pred_norm = np.linalg.norm(pred_w, axis=0, keepdims=True) + 1e-12
    gt_norm = np.linalg.norm(gt_w, axis=0, keepdims=True) + 1e-12
    similarity = np.abs((pred_w.T @ gt_w) / (pred_norm.T @ gt_norm))

    diag_scores = np.clip(np.diag(similarity), 0.0, 1.0)
    best_scores = np.clip(np.max(similarity, axis=0), 0.0, 1.0)

    start = 1 if similarity.shape[0] > 1 else 0
    diag_scores_nontrivial = diag_scores[start:]
    best_scores_nontrivial = best_scores[start:]

    overlap = np.linalg.norm(pred_w.T @ gt_w, ord="fro") ** 2 / max(pred.shape[1], 1)
    subspace_projection_error = float(max(0.0, 1.0 - overlap))

    eval_relerr = np.abs(pred_eval - gt_eval) / (np.abs(gt_eval) + 1e-8)
    if eval_relerr.size > 0:
        eval_relerr[0] = 0.0
    eval_relerr_nontrivial = eval_relerr[start:]

    return {
        "case": case.name,
        "split": case.split,
        "n_vertices": int(case.vertices.shape[0]),
        "k_process": int(pred.shape[1]),
        "diag_abs_cosine_mean": float(np.mean(diag_scores)),
        "diag_abs_cosine_min": float(np.min(diag_scores)),
        "diag_abs_cosine_mean_nontrivial": float(np.mean(diag_scores_nontrivial)),
        "diag_abs_cosine_min_nontrivial": float(np.min(diag_scores_nontrivial)),
        "best_match_abs_cosine_mean_nontrivial": float(np.mean(best_scores_nontrivial)),
        "best_match_abs_cosine_min_nontrivial": float(np.min(best_scores_nontrivial)),
        "subspace_projection_error": subspace_projection_error,
        "subspace_overlap": float(max(0.0, 1.0 - subspace_projection_error)),
        "eval_relerr_mean_nontrivial": float(np.mean(eval_relerr_nontrivial)),
        "eval_relerr_median_nontrivial": float(np.median(eval_relerr_nontrivial)),
        "eval_relerr_max_nontrivial": float(np.max(eval_relerr_nontrivial)),
    }


def _run_matching_pipeline(mesh1: TriMesh, mesh2: TriMesh, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.signature == "WKS":
        descr1 = mesh_WKS(mesh1, num_E=args.wks_num_E, k=args.wks_k)
        descr2 = mesh_WKS(mesh2, num_E=args.wks_num_E, k=args.wks_k)
    else:
        descr1 = mesh_HKS(mesh1, num_T=args.wks_num_E, k=args.wks_k)
        descr2 = mesh_HKS(mesh2, num_T=args.wks_num_E, k=args.wks_k)

    p2p_init = np.asarray(knn_query(descr1, descr2, k=1), dtype=np.int64).reshape(-1)
    _, p2p_zoomout = mesh_zoomout_refine_p2p(
        p2p_21=p2p_init,
        mesh1=mesh1,
        mesh2=mesh2,
        k_init=args.zo_k_init,
        nit=args.zo_nit,
        step=args.zo_step,
        return_p2p=True,
        n_jobs=args.n_jobs,
        verbose=False,
    )
    return p2p_init, np.asarray(p2p_zoomout, dtype=np.int64).reshape(-1)


def _load_case_from_h5(
    mesh_h5: h5py.File,
    corr_h5: h5py.File,
    index: int,
    split: str,
    model: PretrainTraining,
    k_process: int,
) -> CaseData:
    obj_path = mesh_h5["obj_paths"][index].decode("utf-8")
    name = Path(obj_path).stem
    vertices = _reshape_vlen(mesh_h5["verts"], mesh_h5["vert_shapes"], index, dtype=np.float32)
    faces = _reshape_vlen(mesh_h5["faces"], mesh_h5["face_shapes"], index, dtype=np.int32)
    corres = _load_vlen_array(corr_h5["corres"], index, dtype=np.int64)
    gt_eval_full = _load_vlen_array(corr_h5["evals"], index, dtype=np.float32)
    gt_evec_full = _reshape_vlen(corr_h5["evecs"], corr_h5["evec_shapes"], index, dtype=np.float32)

    if gt_eval_full.shape[0] < k_process or gt_evec_full.shape[1] < k_process:
        raise ValueError(f"{name}: HDF5 only stores {gt_evec_full.shape[1]} eigenvectors, need k_process={k_process}.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{name}: invalid vertex shape {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{name}: invalid face shape {faces.shape}.")
    if corres.ndim != 1:
        raise ValueError(f"{name}: invalid correspondence shape {corres.shape}.")
    if np.min(corres) < 0 or np.max(corres) >= vertices.shape[0]:
        raise ValueError(f"{name}: correspondence indices are out of range for {vertices.shape[0]} vertices.")

    pred_eval, pred_evec, mass = _infer_rr_eigenpairs(model, vertices, faces, k_process=k_process)
    gt_eval = gt_eval_full[:k_process].astype(np.float32)
    gt_evec = gt_evec_full[:, :k_process].astype(np.float32)

    mesh_gt = _build_matching_mesh(vertices, faces, gt_eval, gt_evec)
    mesh_pred = _build_matching_mesh(vertices, faces, pred_eval, pred_evec)
    return CaseData(
        name=name,
        split=split,
        obj_path=obj_path,
        vertices=vertices,
        faces=faces.astype(np.int64),
        corres=corres,
        gt_eval=gt_eval,
        gt_evec=gt_evec,
        pred_eval=pred_eval,
        pred_evec=pred_evec,
        mass=mass.astype(np.float32),
        mesh_gt=mesh_gt,
        mesh_pred=mesh_pred,
    )


def _load_cases(
    data_dir: Path,
    model: PretrainTraining,
    k_process: int,
    case_filter: str | None,
) -> tuple[dict[str, CaseData], list[str]]:
    split_files = [
        ("train", data_dir / "functional_mapping_train_mesh.hdf5", data_dir / "functional_mapping_train_corr.hdf5"),
        ("test", data_dir / "functional_mapping_test_mesh.hdf5", data_dir / "functional_mapping_test_corr.hdf5"),
    ]
    cases: dict[str, CaseData] = {}
    all_case_names: list[str] = []

    for split, mesh_path, corr_path in split_files:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh HDF5: {mesh_path}")
        if not corr_path.exists():
            raise FileNotFoundError(f"Missing corr HDF5: {corr_path}")

        with h5py.File(mesh_path, "r") as mesh_h5, h5py.File(corr_path, "r") as corr_h5:
            count = len(mesh_h5["obj_paths"])
            for index in range(count):
                name = Path(mesh_h5["obj_paths"][index].decode("utf-8")).stem
                all_case_names.append(name)
                if case_filter is not None and not fnmatchcase(name, case_filter):
                    continue
                case = _load_case_from_h5(
                    mesh_h5=mesh_h5,
                    corr_h5=corr_h5,
                    index=index,
                    split=split,
                    model=model,
                    k_process=k_process,
                )
                if case.name in cases:
                    raise ValueError(f"Duplicate case name found across HDF5 splits: {case.name}")
                cases[case.name] = case

    all_case_names = sorted(all_case_names)
    if not cases:
        raise ValueError(f"No case names matched --case-filter={case_filter!r}.")
    return cases, all_case_names


def _split_combo(source: CaseData, target: CaseData) -> str:
    return f"{source.split}->{target.split}"


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {
        "num_pairs": len(rows),
        "metric_aggregation": "Summaries are computed over per-pair scalar metrics from ordered pairs.",
        "metrics": {},
    }
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        summary["metrics"][field] = _summarize_values(values)
    return summary


def _write_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows available for CSV output.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.ckpt, args.device)
    cases, all_case_names = _load_cases(
        args.data_dir,
        model=model,
        k_process=args.k_process,
        case_filter=args.case_filter,
    )
    case_names = sorted(cases)
    if len(case_names) < 2:
        raise ValueError("Need at least two cases after filtering to evaluate pairwise matching.")

    spectral_rows = [_spectral_metrics(cases[name]) for name in case_names]
    spectral_summary = _summarize_rows(spectral_rows)

    total_pairs = len(case_names) * (len(case_names) - 1)
    if args.max_pairs > 0:
        total_pairs = min(total_pairs, args.max_pairs)

    rows: list[dict[str, Any]] = []
    geodesic_cache: dict[str, np.ndarray] = {}
    processed = 0
    progress = tqdm(total=total_pairs, desc="FAUST pairs")

    try:
        for source_name in case_names:
            source = cases[source_name]
            if source_name not in geodesic_cache:
                geodesic_cache[source_name] = _compute_geodesic_matrix(source.vertices, source.faces)
            geodesic = geodesic_cache[source_name]
            gt_template = source.corres

            for target_name in case_names:
                if source_name == target_name:
                    continue
                target = cases[target_name]
                if source.corres.shape[0] != target.corres.shape[0]:
                    raise ValueError(
                        f"Template correspondence length mismatch: {source.name}={source.corres.shape[0]} vs "
                        f"{target.name}={target.corres.shape[0]}"
                    )

                pred_init, pred_zoomout = _run_matching_pipeline(source.mesh_pred, target.mesh_pred, args)
                gt_init, gt_zoomout = _run_matching_pipeline(source.mesh_gt, target.mesh_gt, args)

                pred_init_sampled = pred_init[target.corres]
                pred_zoomout_sampled = pred_zoomout[target.corres]
                gt_init_sampled = gt_init[target.corres]
                gt_zoomout_sampled = gt_zoomout[target.corres]

                err_pred_template_init = geodesic[gt_template, pred_init_sampled]
                err_pred_template_zoomout = geodesic[gt_template, pred_zoomout_sampled]
                err_pred_gtfm_init = geodesic[gt_init_sampled, pred_init_sampled]
                err_pred_gtfm_zoomout = geodesic[gt_zoomout_sampled, pred_zoomout_sampled]
                err_gt_template_init = geodesic[gt_template, gt_init_sampled]
                err_gt_template_zoomout = geodesic[gt_template, gt_zoomout_sampled]

                row: dict[str, Any] = {
                    "source": source.name,
                    "target": target.name,
                    "source_split": source.split,
                    "target_split": target.split,
                    "split_combo": _split_combo(source, target),
                    "n_source_vertices": int(source.vertices.shape[0]),
                    "n_target_vertices": int(target.vertices.shape[0]),
                    "template_count": int(gt_template.shape[0]),
                    "pred_template_acc_init": float(np.mean(pred_init_sampled == gt_template)),
                    "pred_template_acc_zoomout": float(np.mean(pred_zoomout_sampled == gt_template)),
                    "pred_gtfm_acc_init": float(np.mean(pred_init_sampled == gt_init_sampled)),
                    "pred_gtfm_acc_zoomout": float(np.mean(pred_zoomout_sampled == gt_zoomout_sampled)),
                    "gt_template_acc_init": float(np.mean(gt_init_sampled == gt_template)),
                    "gt_template_acc_zoomout": float(np.mean(gt_zoomout_sampled == gt_template)),
                }
                row.update(_metric_bundle(err_pred_template_init, "pred_template_geod_init"))
                row.update(_metric_bundle(err_pred_template_zoomout, "pred_template_geod_zoomout"))
                row.update(_metric_bundle(err_pred_gtfm_init, "pred_gtfm_geod_init"))
                row.update(_metric_bundle(err_pred_gtfm_zoomout, "pred_gtfm_geod_zoomout"))
                row.update(_metric_bundle(err_gt_template_init, "gt_template_geod_init"))
                row.update(_metric_bundle(err_gt_template_zoomout, "gt_template_geod_zoomout"))

                rows.append(row)
                processed += 1
                progress.update(1)
                if args.max_pairs > 0 and processed >= args.max_pairs:
                    break
            if args.max_pairs > 0 and processed >= args.max_pairs:
                break
    finally:
        progress.close()

    if not rows:
        raise ValueError("No pairs were evaluated.")

    summary = {
        "config": {
            "ckpt": args.ckpt.as_posix(),
            "data_dir": args.data_dir.as_posix(),
            "out_dir": args.out_dir.as_posix(),
            "device": args.device,
            "k_process": int(args.k_process),
            "signature": args.signature,
            "wks_num_E": int(args.wks_num_E),
            "wks_k": int(args.wks_k),
            "zo_k_init": int(args.zo_k_init),
            "zo_nit": int(args.zo_nit),
            "zo_step": int(args.zo_step),
            "n_jobs": int(args.n_jobs),
            "case_filter": args.case_filter,
            "max_pairs": int(args.max_pairs),
        },
        "dataset": {
            "num_cases_total": len(all_case_names),
            "num_cases_evaluated": len(case_names),
            "evaluated_cases": case_names,
            "num_pairs_evaluated": len(rows),
        },
        "spectral": spectral_summary,
        "overall": _summarize_rows(rows),
        "by_split_combo": {},
    }
    for combo in sorted({row["split_combo"] for row in rows}):
        combo_rows = [row for row in rows if row["split_combo"] == combo]
        summary["by_split_combo"][combo] = _summarize_rows(combo_rows)

    summary_path = args.out_dir / args.summary_name
    pairs_path = args.out_dir / args.pairs_name
    spectral_path = args.out_dir / args.spectral_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_pairs_csv(pairs_path, rows)
    _write_pairs_csv(spectral_path, spectral_rows)

    print(f"Loaded {len(cases)} cases from checkpoint {args.ckpt.name}.")
    print(
        "Spectral summary: "
        f"diag_abs_cosine_mean_nontrivial={spectral_summary['metrics']['diag_abs_cosine_mean_nontrivial']['mean']:.6f}, "
        f"best_match_abs_cosine_mean_nontrivial={spectral_summary['metrics']['best_match_abs_cosine_mean_nontrivial']['mean']:.6f}, "
        f"subspace_projection_error={spectral_summary['metrics']['subspace_projection_error']['mean']:.6f}, "
        f"eval_relerr_mean_nontrivial={spectral_summary['metrics']['eval_relerr_mean_nontrivial']['mean']:.6f}"
    )
    print(
        "Matching summary: "
        f"pred_template_geod_zoomout_mean={summary['overall']['metrics']['pred_template_geod_zoomout_mean']['mean']:.6f}, "
        f"pred_gtfm_geod_zoomout_mean={summary['overall']['metrics']['pred_gtfm_geod_zoomout_mean']['mean']:.6f}, "
        f"gt_template_geod_zoomout_mean={summary['overall']['metrics']['gt_template_geod_zoomout_mean']['mean']:.6f}, "
        f"pred_template_acc_zoomout={summary['overall']['metrics']['pred_template_acc_zoomout']['mean']:.6f}"
    )
    print(f"Evaluated {len(rows)} ordered pairs.")
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-pair metrics to {pairs_path}")
    print(f"Saved spectral metrics to {spectral_path}")


if __name__ == "__main__":
    main()
