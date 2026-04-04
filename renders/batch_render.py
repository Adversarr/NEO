import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import colors as mpl_colors
import shutil
import os
import seaborn
import colorcet
import cmcrameri.cm as cmc

@dataclass(frozen=True)
class CasePaths:
    case_dir: Path
    results_json: Path
    gt_source: str
    pred_source: str
    gt_ply: Path
    pred_ply: Path
    err_ply: Path
    gt_png: Path
    pred_png: Path
    err_png: Path


def _iter_cases(root: Path, glob: str):
    for p in sorted(root.glob(glob)):
        if not p.is_dir():
            continue
        results_json = p / "inferred" / "results.json"
        if results_json.exists():
            yield p


def _load_results(case_dir: Path):
    import json

    with (case_dir / "inferred" / "results.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        **meta,
    }


def _load_matrix(case_dir: Path, source: str) -> np.ndarray:
    if source == "mesh":
        path = case_dir / "input" / "mesh_evec.npy"
    elif source == "pointcloud":
        path = case_dir / "inferred" / "pc_evec.npy"
    elif source in ("network", "network_fp32"):
        path = case_dir / "inferred" / "net_evec_fp32.npy"
        if not path.exists():
            path = case_dir / "inferred" / "net_evec.npy"
    elif source == "network_fp16":
        path = case_dir / "inferred" / "net_evec_fp16.npy"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not path.exists():
        raise FileNotFoundError(path)

    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got {mat.shape}")
    return np.asarray(mat, dtype=np.float64)


def _load_points_and_mass(case_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.load(case_dir / "sample_points.npy")
    # points = np.load(case_dir / "input" / "points.npy")
    mass_path = case_dir / "input" / "mass.npy"
    mass = np.load(mass_path) if mass_path.exists() else None
    return np.asarray(points, dtype=np.float64), (None if mass is None else np.asarray(mass, dtype=np.float64).reshape(-1))


def _load_pred_original(case_dir: Path, precision: str = "fp32") -> np.ndarray:
    if precision == "fp32":
        path = case_dir / "inferred" / "net_pred_original_fp32.npy"
        if not path.exists():
            path = case_dir / "inferred" / "net_pred_original.npy"
    elif precision == "fp16":
        path = case_dir / "inferred" / "net_pred_original_fp16.npy"
    else:
        raise ValueError(f"Unknown precision: {precision}")
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 at {path}, got {arr.shape}")
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features at {path}, got {arr.shape}")
    return arr


def _cosine_similarity_abs(gt_vec: np.ndarray, pred_mat: np.ndarray, mass: np.ndarray | None):
    gt = np.asarray(gt_vec, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_mat, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError(f"pred must be 2D, got shape {pred.shape}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    if mass is None:
        gt_norm = np.linalg.norm(gt)
        if gt_norm == 0:
            return np.zeros((pred.shape[1],), dtype=np.float64)
        pred_norms = np.linalg.norm(pred, axis=0)
        denom = pred_norms * gt_norm
        denom = np.where(denom == 0, np.inf, denom)
        dots = pred.T @ gt
        return np.abs(dots / denom)

    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    gt_m = m * gt
    dots = pred.T @ gt_m
    gt_norm = float(np.sqrt(np.dot(gt, gt_m)))
    if gt_norm == 0:
        return np.zeros((pred.shape[1],), dtype=np.float64)
    pred_norms = np.sqrt(np.sum(pred * (m.reshape(-1, 1) * pred), axis=0))
    denom = pred_norms * gt_norm
    denom = np.where(denom == 0, np.inf, denom)
    return np.abs(dots / denom)


def _cosine_similarity(gt_vec: np.ndarray, pred_vec: np.ndarray, mass: np.ndarray | None):
    gt = np.asarray(gt_vec, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_vec, dtype=np.float64).reshape(-1)
    if gt.shape[0] != pred.shape[0]:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    if mass is None:
        gt_norm = np.linalg.norm(gt)
        pred_norm = np.linalg.norm(pred)
        denom = gt_norm * pred_norm
        if denom == 0:
            return 0.0
        return float((gt @ pred) / denom)

    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    gt_m = m * gt
    pred_m = m * pred
    num = float(np.dot(gt, pred_m))
    denom = float(np.sqrt(np.dot(gt, gt_m)) * np.sqrt(np.dot(pred, pred_m)))
    if denom == 0:
        return 0.0
    return num / denom


def _write_colored_ply(path: Path, points: np.ndarray, colors_u8: np.ndarray):
    pts = np.asarray(points)
    col = np.asarray(colors_u8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")
    if col.ndim != 2 or col.shape[1] != 3:
        raise ValueError(f"colors must be (N,3), got {col.shape}")
    if pts.shape[0] != col.shape[0]:
        raise ValueError(f"N mismatch: points {pts.shape[0]} vs colors {col.shape[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {pts.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        for (x, y, z), (r, g, b) in zip(pts, col):
            f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")


def _colormap_colors(values: np.ndarray, cmap_name: str, vmin: float, vmax: float, vcenter: float | None = None):
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.clip(values, vmin, vmax)
    if vcenter is not None and (vmin < vcenter < vmax):
        norm = mpl_colors.TwoSlopeNorm(vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax))
        t = norm(vals)
        t = np.where(np.isfinite(t), t, 0.5)
        t = np.clip(t, 0.0, 1.0)
    else:
        denom = float(vmax - vmin)
        if denom <= 0:
            t = np.full_like(vals, 0.5 if vcenter is not None else 0.0, dtype=np.float64)
        else:
            t = (vals - float(vmin)) / denom
        t = np.clip(t, 0.0, 1.0)

    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except ValueError:
        # Try common prefixes for scientific or colorcet colormaps
        for prefix in ["cmc.", "cet_"]:
            try:
                cmap = matplotlib.colormaps.get_cmap(prefix + cmap_name)
                break
            except ValueError:
                continue
        else:
            raise

    rgba = cmap(t)
    rgb_u8 = (rgba[:, :3] * 255.0).round().astype(np.uint8)
    return rgb_u8


def _build_eigen_case_paths(case_dir: Path, k: int, gt_source: str, pred_source: str):
    render_dir = case_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_ply = render_dir / f"gt_{gt_source}_eigen_{k}.ply"
    pred_ply = render_dir / f"pred_{pred_source}_eigen_{k}.ply"
    err_ply = render_dir / f"err_{pred_source}_vs_{gt_source}_eigen_{k}.ply"
    gt_png = render_dir / f"gt_{gt_source}_eigen_{k}.png"
    pred_png = render_dir / f"pred_{pred_source}_eigen_{k}.png"
    err_png = render_dir / f"err_{pred_source}_vs_{gt_source}_eigen_{k}.png"
    return CasePaths(
        case_dir=case_dir,
        results_json=case_dir / "inferred" / "results.json",
        gt_source=gt_source,
        pred_source=pred_source,
        gt_ply=gt_ply,
        pred_ply=pred_ply,
        err_ply=err_ply,
        gt_png=gt_png,
        pred_png=pred_png,
        err_png=err_png,
    )


def _render_with_blender(
    blender_bin: str,
    blender_script: Path,
    ply_path: Path,
    png_path: Path,
    render_extra_args: list[str],
):
    cmd = [
        blender_bin,
        "-b",
        "-P",
        blender_script.as_posix(),
        "--",
        "--input",
        ply_path.as_posix(),
        "--output",
        png_path.as_posix(),
    ]
    cmd.extend(render_extra_args)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch render eigenvector/feature visualizations")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing case folders")
    parser.add_argument("--k", type=int, nargs="+", default=None, help="Eigen index list (0-based)")
    parser.add_argument("--feat", type=int, nargs="+", default=None, help="Feature index list from net_pred_original.npy")
    parser.add_argument(
        "--feat_precisions",
        type=str,
        nargs="+",
        default=["fp32", "fp16"],
        choices=["fp32", "fp16"],
        help="Precisions to render for features",
    )
    parser.add_argument("--gt", type=str, default="pointcloud", choices=["mesh", "pointcloud"], help="Ground-truth source")
    parser.add_argument("--pred", type=str, nargs="+", default=["network_fp32"], choices=["network", "network_fp32", "network_fp16", "pointcloud", "mesh"], help="Prediction source(s)")
    parser.add_argument("--cmap", type=str, default=None, help="Colormap name override for both eig/feat")
    parser.add_argument("--eig_cmap", type=str, default="viridis", help="Colormap for eigenvectors (0 is neutral)")
    parser.add_argument("--feat_cmap", type=str, default="viridis", help="Colormap for features")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max")
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Force symmetric scale around 0 (eigenvectors are symmetric unless --eig_asymmetric)",
    )
    parser.add_argument("--eig_asymmetric", action="store_true", help="Disable symmetric zero-centered scaling for eigenvectors")
    parser.add_argument("--global_scale", action="store_true", help="Use same vmin/vmax across all cases")

    parser.add_argument("--err_cmap", type=str, default="magma", help="Colormap for error map")
    parser.add_argument("--err_vmax", type=float, default=None, help="Manual error vmax for cmap")
    parser.add_argument("--err_global_vmax", action="store_true", help="Use same error vmax across all cases")
    parser.add_argument("--no_error", action="store_true", help="Disable error-map export/render")
    parser.add_argument("--write_only", action="store_true", help="Only write PLY files")
    parser.add_argument("--render_only", action="store_true", help="Only render existing PLY files")
    parser.add_argument("--blender", type=str, default="blender", help="Blender executable")
    parser.add_argument(
        "--render_script",
        type=str,
        default=(Path(__file__).resolve().parent / "render_once.py").as_posix(),
        help="Blender render script path",
    )
    parser.add_argument("render_extra", nargs=argparse.REMAINDER, help="Extra args passed to Blender render script")
    parser.add_argument("--glob", type=str, default='*', help="Glob pattern to match case folders")
    parser.add_argument("--eigs_scale", type=float, default=1.0, help="Scale factor for eigenvectors")

    args = parser.parse_args()
    g_scal = args.eigs_scale
    if args.cmap is not None:
        args.eig_cmap = args.cmap
        args.feat_cmap = args.cmap
    root = Path(args.root)
    k_list = [] if args.k is None else [int(v) for v in args.k]
    feat_list = [] if args.feat is None else [int(v) for v in args.feat]
    if not k_list and not feat_list:
        k_list = [0]
    if any(k < 0 for k in k_list):
        raise ValueError("All k must be >= 0")
    if any(f < 0 for f in feat_list):
        raise ValueError("All feat must be >= 0")

    case_dirs = list(_iter_cases(root, args.glob))

    if not case_dirs:
        raise FileNotFoundError(f"No case folders found under {root}")

    render_extra_args = list(args.render_extra or [])
    if render_extra_args[:1] == ["--"]:
        render_extra_args = render_extra_args[1:]

    # default_flags = ["--auto_frame", "--auto_lights", "--lights_target_center"]
    
    # Soften shadows by increasing light sizes if not specified by user
    defaults = {
        "--area_light_size": "12.0",
        "--point_light_soft_size": "2.0"
    }
    for flag, val in defaults.items():
        if flag not in render_extra_args:
             render_extra_args.extend([flag, val])
    render_script = Path(args.render_script)

    pred_sources = args.pred if isinstance(args.pred, list) else [str(args.pred)]

    for k in k_list:
        symmetric_eig = args.symmetric or (not args.eig_asymmetric)
        scale_vmin = args.vmin
        scale_vmax = args.vmax
        err_vmax_global = args.err_vmax
        if args.global_scale and (scale_vmin is None or scale_vmax is None) and not args.render_only:
            all_vals = []
            for case_dir in case_dirs:
                _, mass = _load_points_and_mass(case_dir)
                gt = _load_matrix(case_dir, args.gt)
                # Use the first pred source for global scale computation
                pred = _load_matrix(case_dir, pred_sources[0])
                if k >= gt.shape[1]:
                    continue
                gt_k = gt[:, k]
                sim = _cosine_similarity_abs(gt_k, pred, mass)
                best_idx = int(np.argmax(sim))
                pred_k = pred[:, best_idx]
                if _cosine_similarity(gt_k, pred_k, mass) < 0:
                    pred_k = -pred_k
                all_vals.append(gt_k)
                all_vals.append(pred_k)

            if not all_vals:
                raise ValueError(f"No valid cases contain eigen index k={k}")

            stacked = np.concatenate(all_vals, axis=0)
            if symmetric_eig:
                m = float(np.max(np.abs(stacked)))
                scale_vmin, scale_vmax = -m, m
            else:
                scale_vmin, scale_vmax = float(np.min(stacked)), float(np.max(stacked))

        if args.err_global_vmax and err_vmax_global is None and (not args.render_only) and (not args.no_error):
            max_err = 0.0
            any_case = False
            for case_dir in case_dirs:
                _, mass = _load_points_and_mass(case_dir)
                gt = _load_matrix(case_dir, args.gt)
                pred = _load_matrix(case_dir, pred_sources[0])
                if k >= gt.shape[1]:
                    continue
                gt_k = gt[:, k]
                sim = _cosine_similarity_abs(gt_k, pred, mass)
                best_idx = int(np.argmax(sim))
                pred_k = pred[:, best_idx]
                if _cosine_similarity(gt_k, pred_k, mass) < 0:
                    pred_k = -pred_k
                err = np.abs(pred_k - gt_k)
                max_err = max(max_err, float(np.max(err)))
                any_case = True

            if not any_case:
                raise ValueError(f"No valid cases contain eigen index k={k}")
            err_vmax_global = max_err

        for case_dir in case_dirs:
            for pred_src in pred_sources:
                paths = _build_eigen_case_paths(case_dir, k, args.gt, pred_src)

                if not args.render_only:
                    points, mass = _load_points_and_mass(paths.case_dir)
                    gt = _load_matrix(paths.case_dir, args.gt)
                    pred = _load_matrix(paths.case_dir, pred_src)

                    if k >= gt.shape[1]:
                        continue

                    gt_k = gt[:, k]
                    sim = _cosine_similarity_abs(gt_k, pred, mass)
                    best_idx = int(np.argmax(sim))
                    pred_k = pred[:, best_idx]
                    if _cosine_similarity(gt_k, pred_k, mass) < 0:
                        pred_k = -pred_k

                    if not args.no_error:
                        err_k = np.abs(pred_k - gt_k)

                    if scale_vmin is None and scale_vmax is None:
                        # if symmetric_eig:
                        #     # Use 98th percentile for robust scaling
                        #     m = float(np.percentile(np.abs(np.concatenate([gt_k, pred_k], axis=0)), 98))
                        #     vmin, vmax = -m, m
                        # else:
                        #     vmin = float(np.percentile(np.concatenate([gt_k, pred_k]), 2))
                        #     vmax = float(np.percentile(np.concatenate([gt_k, pred_k]), 98))
                        if symmetric_eig:
                            m = float(np.max(np.abs(np.concatenate([gt_k, pred_k], axis=0))))
                            vmin, vmax = -m, m
                        else:
                            vmin = float(min(np.min(gt_k), np.min(pred_k)))
                            vmax = float(max(np.max(gt_k), np.max(pred_k)))
                    else:
                        vmin = scale_vmin
                        vmax = scale_vmax
                        if symmetric_eig:
                            if vmin is None and vmax is not None:
                                vmax = float(vmax)
                                vmin = -abs(vmax)
                            elif vmax is None and vmin is not None:
                                vmin = float(vmin)
                                vmax = abs(vmin)
                            elif vmin is not None and vmax is not None:
                                vmin = float(vmin)
                                vmax = float(vmax)
                                m = max(abs(vmin), abs(vmax))
                                vmin, vmax = -m, m
                        if vmin is None or vmax is None:
                            raise ValueError("vmin/vmax must both be set when not auto-scaling")
                        vmin, vmax = float(vmin), float(vmax)

                    gt_colors = _colormap_colors(g_scal * gt_k, args.eig_cmap, vmin=vmin, vmax=vmax, vcenter=0.0)
                    pred_colors = _colormap_colors(g_scal * pred_k, args.eig_cmap, vmin=vmin, vmax=vmax, vcenter=0.0)

                    _write_colored_ply(paths.gt_ply, points, gt_colors)
                    _write_colored_ply(paths.pred_ply, points, pred_colors)

                    if not args.no_error:
                        err_vmax_case = err_vmax_global
                        if err_vmax_case is None:
                            err_vmax_case = float(np.max(err_k))
                        err_colors = _colormap_colors(err_k, args.err_cmap, vmin=0.0, vmax=float(err_vmax_case))
                        _write_colored_ply(paths.err_ply, points, err_colors)

                if args.write_only:
                    continue

                if not paths.gt_ply.exists() or not paths.pred_ply.exists():
                    continue

                _render_with_blender(args.blender, render_script, paths.gt_ply, paths.gt_png, render_extra_args)
                os.remove(paths.gt_ply)
                _render_with_blender(args.blender, render_script, paths.pred_ply, paths.pred_png, render_extra_args)
                os.remove(paths.pred_ply)

                if not args.no_error and paths.err_ply.exists():
                    _render_with_blender(args.blender, render_script, paths.err_ply, paths.err_png, render_extra_args)
                    os.remove(paths.err_ply)

    for feat in feat_list:
        scale_vmin = args.vmin
        scale_vmax = args.vmax
        if args.global_scale and (scale_vmin is None or scale_vmax is None) and not args.render_only:
            all_vals = []
            for case_dir in case_dirs:
                feats = _load_pred_original(case_dir, precision=args.feat_precisions[0])
                if feat >= feats.shape[1]:
                    continue
                all_vals.append(feats[:, feat])
            if not all_vals:
                raise ValueError(f"No valid cases contain feature index feat={feat}")
            stacked = np.concatenate(all_vals, axis=0)
            if args.symmetric:
                m = float(np.max(np.abs(stacked)))
                scale_vmin, scale_vmax = -m, m
            else:
                scale_vmin, scale_vmax = float(np.min(stacked)), float(np.max(stacked))

        for case_dir in case_dirs:
            for prec in args.feat_precisions:
                render_dir = case_dir / "render"
                render_dir.mkdir(parents=True, exist_ok=True)
                feat_ply = render_dir / f"feat_{prec}_{feat}.ply"
                feat_png = render_dir / f"feat_{prec}_{feat}.png"

                if not args.render_only:
                    points, _ = _load_points_and_mass(case_dir)
                    feats = _load_pred_original(case_dir, precision=prec)
                    if feat >= feats.shape[1]:
                        continue
                    feat_k = feats[:, feat]

                    if scale_vmin is None or scale_vmax is None:
                        if args.symmetric:
                            m = float(np.max(np.abs(feat_k)))
                            vmin, vmax = -m, m
                        else:
                            vmin = float(np.min(feat_k))
                            vmax = float(np.max(feat_k))
                    else:
                        vmin, vmax = float(scale_vmin), float(scale_vmax)

                    feat_colors = _colormap_colors(feat_k, args.feat_cmap, vmin=vmin, vmax=vmax)
                    _write_colored_ply(feat_ply, points, feat_colors)

                if args.write_only:
                    continue
                if not feat_ply.exists():
                    continue
                _render_with_blender(args.blender, render_script, feat_ply, feat_png, render_extra_args)
                os.remove(feat_ply)

if __name__ == "__main__":
    main()
