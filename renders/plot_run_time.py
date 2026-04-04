import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


_FONTS_REGISTERED = False


def _register_extra_fonts():
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
        if not path.exists():
            continue
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
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.constrained_layout.use": True,
        }
    )


def _safe_get_nested(d: dict[str, Any], keys: list[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_data(root_dir: Path, glob_pattern: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_dir in sorted(root_dir.glob(glob_pattern)):
        if not case_dir.is_dir():
            continue
        json_path = case_dir / "inferred" / "results.json"
        if not json_path.exists():
            json_path = case_dir / "results.json"
        if not json_path.exists():
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

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

        times = data.get("times", {})
        precisions = data.get("precisions", {})
        forward_fp32 = _safe_get_nested(
            precisions, ["fp32", "times", "forward"], _safe_get_nested(times, ["forward"], None)
        )
        network_gev_fp32 = _safe_get_nested(
            precisions, ["fp32", "times", "network_gev"], _safe_get_nested(times, ["network_gev"], None)
        )
        forward_fp16 = _safe_get_nested(precisions, ["fp16", "times", "forward"], None)
        network_gev_fp16 = _safe_get_nested(precisions, ["fp16", "times", "network_gev"], None)
        pointcloud_gev = _safe_get_nested(
            precisions, ["fp32", "times", "pointcloud_gev"], _safe_get_nested(times, ["pointcloud_gev"], None)
        )

        results.append(
            {
                "case_dir": case_dir,
                "case_name": case_dir.name,
                "n_points": n_points,
                "forward_fp32": forward_fp32,
                "network_gev_fp32": network_gev_fp32,
                "forward_fp16": forward_fp16,
                "network_gev_fp16": network_gev_fp16,
                "pointcloud_gev": pointcloud_gev,
                "data": data,
            }
        )
    return results


def _group_by_npoints(results: list[dict[str, Any]]):
    grouped: dict[int, list[dict[str, float]]] = {}
    for r in results:
        n = r.get("n_points")
        if n is None:
            continue
        forward_fp32 = r.get("forward_fp32")
        network_gev_fp32 = r.get("network_gev_fp32")
        forward_fp16 = r.get("forward_fp16")
        network_gev_fp16 = r.get("network_gev_fp16")
        pointcloud_gev = r.get("pointcloud_gev")
        if pointcloud_gev is None:
            continue
        row: dict[str, float] = {}
        if forward_fp32 is not None:
            row["forward_fp32"] = float(forward_fp32)
        if network_gev_fp32 is not None:
            row["network_gev_fp32"] = float(network_gev_fp32)
        if forward_fp16 is not None:
            row["forward_fp16"] = float(forward_fp16)
        if network_gev_fp16 is not None:
            row["network_gev_fp16"] = float(network_gev_fp16)
        row["pointcloud_gev"] = float(pointcloud_gev)
        if forward_fp32 is not None and network_gev_fp32 is not None:
            row["network_total_fp32"] = float(forward_fp32) + float(network_gev_fp32)
        if forward_fp16 is not None and network_gev_fp16 is not None:
            row["network_total_fp16"] = float(forward_fp16) + float(network_gev_fp16)
        grouped.setdefault(int(n), []).append(row)
    return grouped


def _aggregate(grouped: dict[int, list[dict[str, float]]], stat: str):
    n_points = np.array(sorted(grouped.keys()), dtype=np.int64)
    series_names = [
        "pointcloud_gev",
        "network_gev_fp32",
        "network_gev_fp16",
        "forward_fp32",
        "forward_fp16",
        "network_total_fp32",
        "network_total_fp16",
    ]
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}

    for name in series_names:
        per_n = []
        for n in n_points:
            vals = [row[name] for row in grouped[int(n)] if name in row]
            if not vals:
                arr = np.array([np.nan], dtype=np.float64)
            else:
                arr = np.array(vals, dtype=np.float64)
            per_n.append(arr)

        if stat == "median":
            means[name] = np.array([float(np.median(v)) for v in per_n], dtype=np.float64)
            q1 = np.array([float(np.percentile(v, 25)) for v in per_n], dtype=np.float64)
            q3 = np.array([float(np.percentile(v, 75)) for v in per_n], dtype=np.float64)
            stds[name] = (q3 - q1) / 2.0
        else:
            means[name] = np.array([float(np.mean(v)) for v in per_n], dtype=np.float64)
            stds[name] = np.array([float(np.std(v)) for v in per_n], dtype=np.float64)

    return n_points, means, stds


def plot_run_time(results: list[dict[str, Any]], args):
    grouped = _group_by_npoints(results)
    if not grouped:
        print("No valid results.json entries found with required time fields.")
        return

    setup_style(args)
    n_points, means, stds = _aggregate(grouped, args.stat)

    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize), constrained_layout=True)
    ax_left, ax_right = axes[0], axes[1]

    colors = plt.cm.tab10.colors
    left_series = [
        ("pointcloud_gev", "ARPACK", colors[0]),
        ("network_total_fp32", "Network Total (fp32)", colors[1]),
        ("network_total_fp16", "Network Total (fp16)", colors[3]),
    ]

    for i, (key, label, color) in enumerate(left_series):
        y = means[key]
        yerr = stds[key]

        if args.show_individual:
            xs = []
            ys = []
            for n in n_points:
                for row in grouped[int(n)]:
                    xs.append(int(n))
                    ys.append(float(row[key]))
            ax_left.scatter(xs, ys, s=10, alpha=0.15, color=color, zorder=1)

        ax_left.plot(
            n_points,
            y,
            marker="o" if args.show_markers else None,
            label=label,
            color=color,
            zorder=3 + i,
        )
        if args.show_std:
            lower = np.maximum(y - yerr, 1e-12)
            upper = np.maximum(y + yerr, 1e-12)
            ax_left.fill_between(n_points, lower, upper, color=color, alpha=0.18, zorder=2)

    ax_left.set_xlabel("Number of Points")
    ax_left.set_ylabel("Solve Time (s)")
    ax_left.set_yscale("log")
    ax_left.set_title("Solver Time vs Sampling Resolution")
    ax_left.set_xticks(n_points)
    ax_left.legend(loc="best", frameon=True)

    bar_width = float(args.bar_width)
    x = np.arange(n_points.shape[0], dtype=np.float64)
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float64) * bar_width

    series_bars = [
        ("forward_fp32", "Forward (fp32)", colors[2]),
        ("forward_fp16", "Forward (fp16)", colors[4]),
        ("network_gev_fp32", "Network GEV", colors[1]),
    ]
    for off, (key, label, color) in zip(offsets, series_bars):
        y = means.get(key, np.zeros_like(n_points, dtype=np.float64))
        yerr = stds.get(key, np.zeros_like(n_points, dtype=np.float64))
        rects = ax_right.bar(
            x + off,
            y,
            width=bar_width,
            yerr=yerr if args.show_std else None,
            capsize=2.5 if args.show_std else 0,
            label=label,
            color=color,
            alpha=0.9,
            zorder=3,
        )
        ax_right.bar_label(rects, padding=3, fmt="%.3f", fontsize=8)

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([str(int(n)) for n in n_points])
    ax_right.set_xlabel("Number of Points")
    ax_right.set_ylabel("Time (s)")
    ax_right.set_title("Forward & Subspace-Solve Breakdown")
    ax_right.legend(loc="best", frameon=True)

    if args.title:
        fig.suptitle(args.title, fontsize=float(args.suptitle_size))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if args.show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot runtime statistics from infer.py results.json")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing case folders.")
    parser.add_argument("--glob", type=str, default="**/*", help="Glob pattern to match case directories.")
    parser.add_argument(
        "--sort_by", type=str, default="n_points", choices=["n_points", "name", "none"], help="Sort cases."
    )
    parser.add_argument("--sort_desc", action="store_true", help="Sort in descending order.")

    parser.add_argument("--output", type=str, default="renders/run_time.pdf", help="Output file path (pdf, png, svg).")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output image.")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12.0, 4.0], help="Figure size (w h) in inches.")
    parser.add_argument("--title", type=str, default=None, help="Figure suptitle.")
    parser.add_argument("--suptitle_size", type=float, default=13.0, help="Suptitle font size.")

    parser.add_argument("--stat", type=str, default="mean", choices=["mean", "median"], help="Aggregation statistic.")
    parser.add_argument("--show_std", action="store_true", default=True, help="Show spread as error bars/bands.")
    parser.add_argument("--no_std", action="store_false", dest="show_std", help="Hide error bars/bands.")
    parser.add_argument("--show_individual", action="store_true", help="Scatter individual case points on left plot.")
    parser.add_argument("--show_markers", action="store_true", help="Show line markers on the left plot.")
    parser.add_argument("--bar_width", type=float, default=0.28, help="Bar width on right subplot.")

    parser.add_argument("--font", type=str, default="libertine", help="Preferred serif font family (e.g., libertine).")
    parser.add_argument("--show", action="store_true", help="Show plot window.")

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

    plot_run_time(results, args)


if __name__ == "__main__":
    main()
