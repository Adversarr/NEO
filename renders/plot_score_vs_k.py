import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from typing import List, Dict, Any, Tuple

_FONTS_REGISTERED = False


def _register_extra_fonts():
    """Register extra fonts like Linux Libertine if found on system."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    extra_dirs = [
        "/usr/share/fonts/opentype/linux-libertine",
        "/usr/share/fonts/truetype/linux-libertine",
        "/usr/local/share/fonts",
        "~/.local/share/fonts",
    ]
    for d in extra_dirs:
        path = Path(d).expanduser()
        if path.exists():
            for ext in ["*.otf", "*.ttf"]:
                for f in path.glob(ext):
                    try:
                        font_manager.fontManager.addfont(str(f))
                    except Exception:
                        pass
    _FONTS_REGISTERED = True


def _pick_serif_fonts(preferred: str) -> list[str]:
    _register_extra_fonts()
    available = {f.name for f in font_manager.fontManager.ttflist}

    fallback = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
    pref = (preferred or "").strip()
    if not pref:
        return fallback

    if "libertine" in pref.lower():
        candidates = ["Linux Libertine O", "Linux Libertine", "Libertine", "Libertinus Serif"]
    else:
        candidates = [pref]

    picked = [name for name in candidates if name in available]
    if picked:
        res = picked + [f for f in fallback if f not in picked]
        print(f"Font picked: {res[0]}")
        return res

    print(f"Preferred font '{pref}' not found. Falling back to: {fallback[0]}")
    return fallback


def setup_style(args):
    """Set up SIGGRAPH-like professional plotting style."""
    plt.style.use("seaborn-v0_8-paper")
    serif_fonts = _pick_serif_fonts(getattr(args, "font", ""))
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.constrained_layout.use": True,
        }
    )


def _metric_transform(scores: np.ndarray, metric: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if metric == "cosine":
        return s
    if metric == "mse":
        val = 1.0 - s
        val = np.maximum(val, 1e-15)
        return val
    raise ValueError(f"Unknown metric: {metric}")


def load_data(root_dir: Path, glob_pattern: str) -> List[Dict[str, Any]]:
    """Load results.json from all matching directories."""
    results = []
    for case_dir in sorted(root_dir.glob(glob_pattern)):
        if not case_dir.is_dir():
            continue
        json_path = case_dir / "inferred" / "results.json"
        if not json_path.exists():
            json_path = case_dir / "results.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    n_points = data.get("n_points")
                    if n_points is None:
                        points_path = case_dir / "input" / "points.npy"
                        if not points_path.exists():
                            points_path = case_dir / "points.npy"
                        if points_path.exists():
                            try:
                                pts = np.load(points_path, mmap_mode="r")
                                n_points = int(pts.shape[0])
                            except Exception:
                                n_points = None
                    results.append(
                        {
                            "case_dir": case_dir,
                            "case_name": case_dir.name,
                            "n_points": n_points,
                            "data": data,
                        }
                    )
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
    return results


def _comparison_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    comparison_keys = {
        "mesh_net": "mesh_vs_network",
        "mesh_pc": "mesh_vs_pointcloud",
        "pc_net": "pointcloud_vs_network",
        "mesh_net_fp32": "mesh_vs_network",
        "mesh_net_fp16": "mesh_vs_network",
        "pc_net_fp32": "pointcloud_vs_network",
        "pc_net_fp16": "pointcloud_vs_network",
    }

    labels = {
        "mesh_net": "Mesh vs Network",
        "mesh_pc": "Mesh vs ARPACK",
        "pc_net": "ARPACK vs Network",
        "mesh_net_fp32": "Mesh vs Network (fp32)",
        "mesh_net_fp16": "Mesh vs Network (fp16)",
        "pc_net_fp32": "ARPACK vs Network (fp32)",
        "pc_net_fp16": "ARPACK vs Network (fp16)",
    }
    return comparison_keys, labels


def _robust_linear_fit(x, y):
    """Robust linear regression (log(y) vs x) for exponential decay."""
    # Use log(y) vs x for fitting
    lx = np.asarray(x, dtype=np.float64)
    ly = np.log(np.asarray(y, dtype=np.float64))

    # Filter out non-finite values
    mask = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[mask], ly[mask]
    if len(lx) < 2:
        return 0.0, 0.0, 0.0

    slope, intercept = np.polyfit(lx, ly, 1)

    # Calculate R-squared
    y_pred = slope * lx + intercept
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return slope, intercept, r_squared


def plot_score_vs_k_aggregate(results: List[Dict[str, Any]], args):
    if not results:
        print("No data found to plot.")
        return

    setup_style(args)
    fig, ax = plt.subplots(figsize=args.figsize, constrained_layout=True)

    comparison_keys, labels = _comparison_maps()
    colors = plt.cm.tab10.colors

    per_comp = {}
    for comp in args.comparisons:
        key = comparison_keys.get(comp)
        if not key:
            continue
        all_scores = []
        for res in results:
            if comp.endswith("_fp32"):
                scores = res["data"].get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)
            elif comp.endswith("_fp16"):
                scores = res["data"].get("precisions", {}).get("fp16", {}).get("scores", {}).get(key)
            else:
                scores = res["data"].get("scores", {}).get(key)
                if scores is None:
                    # Fallback to fp32 precision if not in root
                    scores = res["data"].get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)
            if scores is not None:
                all_scores.append(np.asarray(scores, dtype=np.float64))
        if not all_scores:
            continue
        min_k = int(min(s.shape[0] for s in all_scores))
        scores_arr = np.stack([s[:min_k] for s in all_scores], axis=0)
        scores_arr = _metric_transform(scores_arr, args.metric)
        per_comp[comp] = scores_arr

    if not per_comp:
        print("No matching score arrays found to plot.")
        return

    for i, (comp, scores_arr) in enumerate(per_comp.items()):
        k_vals = np.arange(1, scores_arr.shape[1] + 1)
        mean_scores = np.mean(scores_arr, axis=0)
        std_scores = np.std(scores_arr, axis=0)
        color = colors[i % len(colors)]

        # Plot individual lines if requested
        if args.show_individual:
            for j in range(scores_arr.shape[0]):
                ax.plot(k_vals, scores_arr[j], color=color, alpha=0.1, linewidth=0.5, zorder=1)

        (line,) = ax.plot(
            k_vals,
            mean_scores,
            label=labels.get(comp, comp),
            color=color,
            marker="o" if args.show_markers else None,
            zorder=3,
        )

        if args.show_std:
            lower = mean_scores - std_scores
            upper = mean_scores + std_scores
            if args.log_scale:
                lower = np.maximum(lower, 1e-15)
            ax.fill_between(k_vals, lower, upper, color=line.get_color(), alpha=0.2, zorder=2)

        if args.log_scale and args.fit_slope and args.metric == "mse":
            slope, intercept, r2 = _robust_linear_fit(k_vals, mean_scores)
            y_fit = np.exp(slope * k_vals + intercept)
            ax.plot(
                k_vals,
                y_fit,
                color="gray",
                linestyle="--",
                alpha=0.8,
                label=f"Fit ({comp}): α={slope:.2f}, $R^2$={r2:.2f}",
                zorder=4,
            )
            print(f"Aggregate [{comp}] Fit: slope={slope:.4f}, intercept={intercept:.4f}, R2={r2:.4f}")

    ax.set_xlabel("Eigenvalue Index ($k$)")
    ax.set_ylabel("Score (Cosine Similarity)" if args.metric == "cosine" else "MSE (1 - Cosine Similarity)")

    if args.title:
        ax.set_title(args.title)

    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    else:
        if args.metric == "cosine":
            ax.set_ylim(0.8, 1.0)
        else:
            if args.log_scale:
                ax.set_ylim(1e-6, 0.2)
            else:
                ax.set_ylim(0.0, 0.2)

    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    else:
        # Default: start at 0, end at max_k + extra
        max_k = scores_arr.shape[1]
        ax.set_xlim(0, max_k + args.extra_boundary)

    if args.log_scale:
        ax.set_yscale("log")

    ax.legend(loc="best", frameon=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if args.show:
        plt.show()


def plot_score_vs_k_grid(results: List[Dict[str, Any]], args):
    if not results:
        print("No data found to plot.")
        return

    setup_style(args)

    if args.max_cases is not None:
        results = results[: max(0, int(args.max_cases))]

    n_cases = len(results)
    if n_cases == 0:
        print("No data found to plot.")
        return

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(n_cases / ncols))

    if args.figsize is None:
        fig_w = float(args.cell_size[0]) * ncols
        fig_h = float(args.cell_size[1]) * nrows
        figsize = (fig_w, fig_h)
    else:
        figsize = tuple(args.figsize)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=args.sharex,
        sharey=args.sharey,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(-1)

    comparison_keys, labels = _comparison_maps()
    colors = plt.cm.tab10.colors

    legend_handles = None
    legend_labels = None

    for idx, res in enumerate(results):
        ax = axes_arr[idx]

        # Track scores for fp32 vs fp16 error calculation
        scores_by_precision = {}  # precision_key -> score_array

        for j, comp in enumerate(args.comparisons):
            key = comparison_keys.get(comp)
            if not key:
                continue
            if comp.endswith("_fp32"):
                scores = res["data"].get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)
            elif comp.endswith("_fp16"):
                scores = res["data"].get("precisions", {}).get("fp16", {}).get("scores", {}).get(key)
            else:
                scores = res["data"].get("scores", {}).get(key)
                if scores is None:
                    # Fallback to fp32 precision if not in root
                    scores = res["data"].get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)

            if scores is None:
                continue

            s = _metric_transform(np.asarray(scores, dtype=np.float64), args.metric)
            scores_by_precision[comp] = s
            k_vals = np.arange(1, s.shape[0] + 1)
            ax.plot(
                k_vals,
                s,
                label=labels.get(comp, comp),
                color=colors[j % len(colors)],
                marker="o" if args.show_markers else None,
                alpha=float(args.line_alpha),
            )

            if args.log_scale and args.fit_slope and args.metric == "mse":
                if "fp16" not in comp:
                    slope, intercept, r2 = _robust_linear_fit(k_vals, s)
                    y_fit = np.exp(slope * k_vals + intercept)
                    ax.plot(k_vals, y_fit, color="gray", linestyle="--", alpha=0.6, zorder=4)

                    # Add slope text to the plot (top-left corner)
                    text_str = f"$\\alpha={slope:.2f}$\n$R^2={r2:.2f}$"
                    ax.text(
                        0.05,
                        0.92,
                        text_str,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
                        zorder=5,
                    )

                    print(f"Case [{res['case_name']}][{comp}] Fit: slope={slope:.4f}, R2={r2:.4f}")

        # Calculate and display fp32 vs fp16 error in the bottom-right corner
        # Look for pairs like 'pc_net_fp32' and 'pc_net_fp16'
        displayed_diffs = []
        for comp_name, s_val in scores_by_precision.items():
            if comp_name.endswith("_fp32"):
                base = comp_name[:-5]
                fp16_name = base + "_fp16"
                if fp16_name in scores_by_precision:
                    s_fp32 = s_val
                    s_fp16 = scores_by_precision[fp16_name]
                    # Compute absolute difference
                    diff = np.abs(s_fp32 - s_fp16)
                    mean_diff = np.mean(diff)
                    median_diff = np.median(diff)

                    label = "FP32 vs FP16"
                    displayed_diffs.append(
                        f"{label}:\n$\\Delta$ Mean: {mean_diff:.2e}\n$\\Delta$ Med: {median_diff:.2e}"
                    )

        if displayed_diffs:
            ax.text(
                0.95,
                0.05,
                "\n".join(displayed_diffs),
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
                zorder=5,
            )

        if args.case_title != "none":
            if args.case_title == "name":
                if args.title_mode == "parsed":
                    # Parse the case name: e.g., armadillo_2048 -> Armadillo, N=2048
                    parts = res["case_name"].split("_")
                    name_parts = []
                    for p in parts:
                        if not p.isdigit():
                            name_parts.append(p.capitalize())

                    base_name = " ".join(name_parts)
                    n_points = res.get("n_points")
                    if n_points is not None:
                        title = f"{base_name}, N={n_points}" if base_name else f"N={n_points}"
                    else:
                        title = base_name if base_name else res["case_name"]
                elif args.title_mode == "n_only":
                    n_points = res.get("n_points")
                    title = f"N={n_points}" if n_points is not None else res["case_name"]
                else:  # raw
                    title = res["case_name"]
            else:
                title = str(idx)
            ax.set_title(title, fontsize=float(args.case_title_size))

        if args.ylim is not None:
            ax.set_ylim(args.ylim)
        else:
            if args.metric == "cosine":
                ax.set_ylim(0.8, 1.0)
            else:
                if args.log_scale:
                    ax.set_ylim(1e-6, 0.2)
                else:
                    ax.set_ylim(0.0, 0.2)

        if args.log_scale:
            ax.set_yscale("log")

        if args.xlim is not None:
            ax.set_xlim(args.xlim)
        else:
            # Default: start at 0, end at max_k + extra
            # For grid mode, we use the length of the current scores s
            ax.set_xlim(0, s.shape[0] + args.extra_boundary)

        if (legend_handles is None) or (not legend_handles):
            h, labels_list = ax.get_legend_handles_labels()
            if h:
                legend_handles, legend_labels = h, labels_list

    for ax in axes_arr[n_cases:]:
        ax.set_visible(False)

    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=float(args.suptitle_size))

    for r in range(nrows):
        for c in range(ncols):
            k = r * ncols + c
            if k >= n_cases:
                continue
            ax = axes_arr[k]
            if not args.sharex or r == nrows - 1:
                ax.set_xlabel("Eigenvalue Index ($k$)")
            if not args.sharey or c == 0:
                ax.set_ylabel("Cosine Similarity" if args.metric == "cosine" else "1 - Cosine Similarity")

    if legend_handles and args.legend != "none":
        if args.legend == "top":
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.98 if args.suptitle else 0.92),
                bbox_transform=fig.transFigure,
                ncol=len(legend_labels),
                frameon=False,
                borderaxespad=0,
            )
        elif args.legend == "bottom":
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                bbox_transform=fig.transFigure,
                ncol=len(legend_labels),
                frameon=False,
                borderaxespad=0,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if args.show:
        plt.show()


def plot_score_vs_k(results: List[Dict[str, Any]], args):
    """Generate the score vs k plot."""
    if not results:
        print("No data found to plot.")
        return

    if args.mode == "aggregate":
        plot_score_vs_k_aggregate(results, args)
    else:
        plot_score_vs_k_grid(results, args)


def main():
    parser = argparse.ArgumentParser(description="Plot Score vs K for G2PT results.")

    # Data options
    parser.add_argument(
        "--input_dir", type=str, default="exp/pretrain/results", help="Root directory containing experiment results."
    )
    parser.add_argument("--glob", type=str, default="*", help="Glob pattern to match case directories.")
    parser.add_argument(
        "--sort_by",
        type=str,
        default="n_points",
        choices=["n_points", "name", "none"],
        help="Sort cases before plotting.",
    )
    parser.add_argument("--sort_desc", action="store_true", help="Sort cases in descending order.")

    # Plotting options
    parser.add_argument(
        "--output", type=str, default="renders/score_vs_k.pdf", help="Output file path (pdf, png, svg)."
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the output image.")
    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "aggregate"],
        help="Plot per-case grid or aggregate mean/std.",
    )
    parser.add_argument(
        "--metric", type=str, default="mse", choices=["cosine", "mse"], help="Metric: cosine or mse (1 - cosine)."
    )
    parser.add_argument("--figsize", type=float, nargs=2, default=None, help="Figure size in inches (width height).")
    parser.add_argument(
        "--cell_size", type=float, nargs=2, default=[3.2, 2.6], help="Subplot cell size in inches (w h) for grid mode."
    )
    parser.add_argument("--title", type=str, default=None, help="Title for aggregate plot.")
    parser.add_argument("--suptitle", type=str, default=None, help="Figure title for grid mode.")
    parser.add_argument("--ylim", type=float, nargs=2, default=[1e-4, 1.0], help="Y-axis limits.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="X-axis limits.")
    parser.add_argument(
        "--extra_boundary", type=float, default=0.0, help="Extra space to add to the right of the X-axis."
    )
    parser.add_argument(
        "--comparisons",
        "--comparison",
        type=str,
        nargs="+",
        default=["pc_net_fp32", "pc_net_fp16"],
        choices=["mesh_net", "mesh_pc", "pc_net", "mesh_net_fp32", "mesh_net_fp16", "pc_net_fp32", "pc_net_fp16"],
        help="Which comparisons to plot.",
    )

    # Visual toggles
    parser.add_argument("--show_std", action="store_true", default=True, help="Show standard deviation as shaded area.")
    parser.add_argument("--no_std", action="store_false", dest="show_std", help="Hide standard deviation.")
    parser.add_argument(
        "--show_individual", action="store_true", help="Show individual case lines with low alpha in aggregate mode."
    )
    parser.add_argument("--show_markers", action="store_true", help="Show markers on the lines.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")

    # Grid options
    parser.add_argument("--ncols", type=int, default=2, help="Number of subplot columns in grid mode.")
    parser.add_argument("--max_cases", type=int, default=4, help="Limit number of cases plotted in grid mode.")
    parser.add_argument("--sharex", action="store_true", default=True, help="Share x-axis in grid mode.")
    parser.add_argument("--no_sharex", action="store_false", dest="sharex", help="Do not share x-axis in grid mode.")
    parser.add_argument("--sharey", action="store_true", default=True, help="Share y-axis in grid mode.")
    parser.add_argument("--no_sharey", action="store_false", dest="sharey", help="Do not share y-axis in grid mode.")
    parser.add_argument(
        "--legend", type=str, default="bottom", choices=["top", "bottom", "none"], help="Legend position in grid mode."
    )
    parser.add_argument(
        "--case_title", type=str, default="name", choices=["name", "index", "none"], help="Subplot title per case."
    )
    parser.add_argument(
        "--title_mode",
        type=str,
        default="parsed",
        choices=["raw", "parsed", "n_only"],
        help="How to format the case title.",
    )
    parser.add_argument("--case_title_size", type=float, default=9.5, help="Case title font size.")
    parser.add_argument("--suptitle_size", type=float, default=13.0, help="Figure title font size.")
    parser.add_argument("--line_alpha", type=float, default=0.95, help="Line alpha in grid mode.")
    parser.add_argument("--log_scale", action="store_true", default=True, help="Use log scale for Y-axis.")
    parser.add_argument(
        "--fit_slope", action="store_true", default=True, help="Fit and plot slope for MSE in log-log scale."
    )
    parser.add_argument("--font", type=str, default="libertine", help="Preferred serif font family (e.g., libertine).")

    args = parser.parse_args()

    results = load_data(Path(args.input_dir), args.glob)
    if args.sort_by == "n_points":
        results = sorted(
            results,
            key=lambda r: (
                r.get("n_points") is None,
                int(r.get("n_points") or 0),
                str(r.get("case_name") or ""),
            ),
            reverse=bool(args.sort_desc),
        )
    elif args.sort_by == "name":
        results = sorted(results, key=lambda r: str(r.get("case_name") or ""), reverse=bool(args.sort_desc))
    plot_score_vs_k(results, args)


if __name__ == "__main__":
    main()
