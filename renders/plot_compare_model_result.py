import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from typing import List, Dict, Any

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
    if not root_dir.exists():
        return results

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


def get_fp32_scores(data: Dict[str, Any], key: str = "pointcloud_vs_network") -> np.ndarray:
    """Helper to extract fp32 scores from result data."""
    scores = data.get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)
    if scores is None:
        # Fallback to root if not in precisions
        scores = data.get("scores", {}).get(key)
        # Double check nested fp32 if root is empty but structure might be mixed
        if scores is None:
            scores = data.get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)

    if scores is not None:
        return np.asarray(scores, dtype=np.float64)
    return None


def parse_title(case_name: str, n_points: int, mode: str = "parsed") -> str:
    if mode == "parsed":
        parts = case_name.split("_")
        name_parts = []
        for p in parts:
            if not p.isdigit():
                name_parts.append(p.capitalize())

        base_name = " ".join(name_parts)
        if n_points is not None:
            return f"{base_name}, N={n_points}" if base_name else f"N={n_points}"
        return base_name if base_name else case_name
    elif mode == "n_only":
        return f"N={n_points}" if n_points is not None else case_name
    else:  # raw
        return case_name


def plot_compare_models(model1_results: List[Dict[str, Any]], model2_results: List[Dict[str, Any]], args):
    if not model1_results or not model2_results:
        print("No data found for one or both models.")
        return

    setup_style(args)

    # Prepare data
    # We assume both models have the same cases (same folder names)
    # Match them by case_name
    model1_map = {r["case_name"]: r for r in model1_results}
    model2_map = {r["case_name"]: r for r in model2_results}

    common_cases = sorted(list(set(model1_map.keys()) & set(model2_map.keys())))
    if not common_cases:
        print("No common cases found between models.")
        return

    # Sort cases by n_points descending to pick top ones
    case_n_points = []
    for c in common_cases:
        n = model1_map[c].get("n_points") or 0
        case_n_points.append((c, n))

    # Sort descending by N
    case_n_points.sort(key=lambda x: x[1], reverse=True)

    # Determine grid layout
    nrows, ncols = args.nrows, args.ncols
    total_subplots = nrows * ncols
    n_individual_plots = total_subplots - 1  # Last one is summary

    cases_to_plot = case_n_points[:n_individual_plots]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=tuple(args.figsize) if args.figsize else (ncols * 4, nrows * 3),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(-1)

    colors = plt.cm.tab10.colors
    model1_color = colors[0]
    model2_color = colors[1]

    model1_label = args.model1_name
    model2_label = args.model2_name

    # 1. Plot individual cases (MSE vs K)
    for idx, (case_name, n_points) in enumerate(cases_to_plot):
        ax = axes_arr[idx]

        res1 = model1_map[case_name]
        res2 = model2_map[case_name]

        s1 = get_fp32_scores(res1["data"])
        s2 = get_fp32_scores(res2["data"])

        if s1 is not None:
            mse1 = _metric_transform(s1, "mse")
            k_vals1 = np.arange(1, mse1.shape[0] + 1)
            ax.plot(k_vals1, mse1, color=model1_color, label=model1_label, alpha=0.8)

        if s2 is not None:
            mse2 = _metric_transform(s2, "mse")
            k_vals2 = np.arange(1, mse2.shape[0] + 1)
            ax.plot(k_vals2, mse2, color=model2_color, label=model2_label, alpha=0.8)

        title = parse_title(case_name, n_points, args.title_mode)
        ax.set_title(title, fontsize=10)

        if args.log_scale:
            ax.set_yscale("log")
            ax.set_ylim(args.ylim)

        if idx % ncols == 0:
            ax.set_ylabel("MSE (1 - Cosine)")
        if idx >= total_subplots - ncols:
            ax.set_xlabel("Eigenvalue Index ($k$)")

        # Add simple legend to first plot
        if idx == 0:
            ax.legend(loc="best", frameon=True, fontsize=8)

    # 2. Plot Summary (Mean MSE vs N)
    summary_ax = axes_arr[-1]

    # User asked: "No need to display mean and std content that is not shown in other plots"
    # This means summary plot should ONLY include the cases that were actually plotted in individual subplots.
    summary_cases = [c for c, n in cases_to_plot]

    def collect_stats(model_map, cases):
        stats = {}  # N -> list of all mse values (from all k, all cases)
        for c in cases:
            res = model_map[c]
            n = res.get("n_points")
            if n is None:
                continue
            s = get_fp32_scores(res["data"])
            if s is not None:
                mse = _metric_transform(s, "mse")
                if n not in stats:
                    stats[n] = []
                stats[n].append(mse)

        # Flatten lists
        for n in stats:
            stats[n] = np.concatenate(stats[n])
        return stats

    stats1 = collect_stats(model1_map, summary_cases)
    stats2 = collect_stats(model2_map, summary_cases)

    all_Ns = sorted(list(set(stats1.keys()) | set(stats2.keys())))

    def get_plot_data(stats, Ns, log_scale: bool):
        means = []
        stds = []
        valid_Ns = []
        for n in Ns:
            vals = stats.get(n, None)
            if vals is not None and len(vals) > 0:
                if log_scale:
                    vals = np.log(vals)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_Ns.append(n)
        return np.array(valid_Ns), np.array(means), np.array(stds)

    n1, m1_raw, std1_raw = get_plot_data(stats1, all_Ns, args.log_scale)
    n2, m2_raw, std2_raw = get_plot_data(stats2, all_Ns, args.log_scale)

    # Calculate means for plotting
    if args.log_scale:
        m1 = np.exp(m1_raw)
        m2 = np.exp(m2_raw)
    else:
        m1 = m1_raw
        m2 = m2_raw

    # Plot summary lines with error shading (fill_between)
    summary_ax.plot(n1, m1, "o-", color=model1_color, label=model1_label, linewidth=2, markersize=5)
    summary_ax.plot(n2, m2, "s-", color=model2_color, label=model2_label, linewidth=2, markersize=5)

    if args.errorbar:
        # Re-calculate bounds explicitly here to be sure
        def get_bounds(stats, Ns, log_scale, multiplier):
            lowers = []
            uppers = []
            valid_Ns = []

            for n in Ns:
                vals = stats.get(n, None)
                if vals is not None and len(vals) > 0:
                    if log_scale:
                        # Log space statistics
                        log_vals = np.log(vals)
                        mu = np.mean(log_vals)
                        sigma = np.std(log_vals)
                        lowers.append(np.exp(mu - multiplier * sigma))
                        uppers.append(np.exp(mu + multiplier * sigma))
                    else:
                        # Linear space statistics
                        mu = np.mean(vals)
                        sigma = np.std(vals)
                        lowers.append(mu - multiplier * sigma)
                        uppers.append(mu + multiplier * sigma)
                    valid_Ns.append(n)
            return np.array(valid_Ns), np.array(lowers), np.array(uppers)

        _, l1, u1 = get_bounds(stats1, all_Ns, args.log_scale, args.std_multiplier)
        _, l2, u2 = get_bounds(stats2, all_Ns, args.log_scale, args.std_multiplier)

        summary_ax.fill_between(n1, l1, u1, color=model1_color, alpha=0.2, edgecolor="none")
        summary_ax.fill_between(n2, l2, u2, color=model2_color, alpha=0.2, edgecolor="none")

    summary_ax.set_title("Mean MSE vs Point Count ($N$)", fontsize=10, fontweight="bold")
    summary_ax.set_xlabel("Number of Points ($N$)")
    summary_ax.set_ylabel(f"Mean MSE ($\pm {args.std_multiplier:.1f} \\sigma$)")
    summary_ax.set_xscale("log")
    if args.log_scale:
        summary_ax.set_yscale("log")

    summary_ax.legend(loc="best", frameon=True, fontsize=9, fancybox=True, framealpha=0.8)
    # Enhance grid
    summary_ax.grid(True, which="major", ls="-", alpha=0.4, color="gray")
    summary_ax.grid(True, which="minor", ls=":", alpha=0.2, color="gray")

    # Hide unused subplots if any (between cases and summary)
    for i in range(len(cases_to_plot), total_subplots - 1):
        axes_arr[i].set_visible(False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Comparison plot saved to {output_path}")

    if args.show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare two models' results.")

    parser.add_argument("--dir1", type=str, required=True, help="Directory for model 1 results.")
    parser.add_argument("--dir2", type=str, required=True, help="Directory for model 2 results.")
    parser.add_argument("--output", type=str, default="renders/compare_models.png", help="Output file path.")

    # Model names (default to directory names)
    parser.add_argument("--model1_name", type=str, default=None, help="Name for model 1.")
    parser.add_argument("--model2_name", type=str, default=None, help="Name for model 2.")

    # Layout
    parser.add_argument("--nrows", type=int, default=2, help="Number of rows.")
    parser.add_argument("--ncols", type=int, default=3, help="Number of columns.")
    parser.add_argument("--figsize", type=float, nargs=2, default=None, help="Figure size (w h).")

    # Plotting details
    parser.add_argument("--dpi", type=int, default=300, help="DPI.")
    parser.add_argument("--font", type=str, default="libertine", help="Font family.")
    parser.add_argument("--ylim", type=float, nargs=2, default=[1e-4, 1.0], help="Y-axis limits for individual plots.")
    parser.add_argument("--log_scale", action="store_true", default=True, help="Log scale for Y-axis.")
    parser.add_argument("--std_multiplier", type=float, default=1.0, help="Multiplier for std error bars.")
    parser.add_argument(
        "--errorbar",
        dest="errorbar",
        action="store_true",
        help="Enable std error bars in summary plot.",
    )
    parser.add_argument(
        "--no_errorbar",
        dest="errorbar",
        action="store_false",
        help="Disable std error bars in summary plot.",
    )
    parser.set_defaults(errorbar=True)
    parser.add_argument(
        "--title_mode", type=str, default="parsed", choices=["raw", "parsed", "n_only"], help="Title format."
    )
    parser.add_argument("--show", action="store_true", help="Show plot.")

    args = parser.parse_args()

    # Set default model names if not provided
    if not args.model1_name:
        args.model1_name = Path(args.dir1).name
    if not args.model2_name:
        args.model2_name = Path(args.dir2).name

    results1 = load_data(Path(args.dir1), "*")
    results2 = load_data(Path(args.dir2), "*")

    plot_compare_models(results1, results2, args)


if __name__ == "__main__":
    main()
