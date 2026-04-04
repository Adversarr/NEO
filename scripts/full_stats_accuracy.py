"""Usage:
python3 scripts/full_stats_accuracy.py --data_glob "ldata/time_test/*/*/inferred/results.json"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _summarize(xs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _iter_results_files(project_root: Path, pattern: str) -> List[Path]:
    return list(project_root.glob(pattern))


_FONTS_REGISTERED = False


def _register_extra_fonts() -> None:
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


def _pick_serif_fonts(preferred: str) -> List[str]:
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
        return picked + [f for f in fallback if f not in picked]
    return fallback


@dataclass(frozen=True)
class StyleConfig:
    font: str
    font_size: float
    axes_labelsize: float
    tick_labelsize: float
    legend_fontsize: float
    title_size: float
    line_width: float
    marker_size: float


def setup_style(cfg: StyleConfig) -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

    serif_fonts = _pick_serif_fonts(cfg.font)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "axes.labelsize": cfg.axes_labelsize,
            "font.size": cfg.font_size,
            "legend.fontsize": cfg.legend_fontsize,
            "xtick.labelsize": cfg.tick_labelsize,
            "ytick.labelsize": cfg.tick_labelsize,
            "axes.titlesize": cfg.title_size,
            "lines.linewidth": cfg.line_width,
            "lines.markersize": cfg.marker_size,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "mathtext.fontset": "cm",
        }
    )



def load_aggregated_stats(
    project_root: Path,
    pattern: str,
    precisions: Sequence[str],
 ) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    files = _iter_results_files(project_root, pattern)
    if not files:
        print(f"Warning: no results.json found for pattern: {pattern}")
        return {}

    print(f"Found {len(files)} results.json files for pattern: {pattern}")

    buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    for f in files:
        try:
            content = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
            continue

        if not isinstance(content, Mapping):
            continue

        n_points = content.get("n_points")
        if not isinstance(n_points, int) or n_points <= 0:
            continue

        precisions_obj = content.get("precisions")
        
        def extract_stats_from_obj(obj: Mapping[str, Any]) -> Tuple[float, float, float]:
            scores = obj.get("scores", {})
            if not isinstance(scores, Mapping):
                scores = {}
            
            loss_val = obj.get("loss")
            if loss_val is None:
                loss_val = scores.get("subspace_loss")
            loss = _as_float(loss_val)

            pc_vs_net_list = scores.get("pointcloud_vs_network")
            if pc_vs_net_list is None:
                pc_vs_net_list = scores.get("mesh_vs_network")
            
            if isinstance(pc_vs_net_list, (list, tuple, np.ndarray)) and len(pc_vs_net_list) > 0:
                val = np.mean(pc_vs_net_list)
                pc_vs_net_err = 1.0 - val
            else:
                pc_vs_net_err = float("nan")

            rel_err_list = scores.get("eval_relerr")
            if isinstance(rel_err_list, (list, tuple, np.ndarray)) and len(rel_err_list) > 0:
                rel_err = np.mean(rel_err_list)
            else:
                rel_err = float("nan")

            return float(loss), float(pc_vs_net_err), float(rel_err)

        if isinstance(precisions_obj, Mapping):
            for p in precisions:
                pobj = precisions_obj.get(p)
                if not isinstance(pobj, Mapping):
                    continue
                loss, pc_vs_net_err, rel_err = extract_stats_from_obj(pobj)
                key = (p, int(n_points))
                dst = buckets.setdefault(key, {"subspace_loss": [], "pc_vs_net_err": [], "eval_relerr": []})
                dst["subspace_loss"].append(loss)
                dst["pc_vs_net_err"].append(pc_vs_net_err)
                dst["eval_relerr"].append(rel_err)
        else:
            if "fp32" in precisions:
                loss, pc_vs_net_err, rel_err = extract_stats_from_obj(content)
                key = ("fp32", int(n_points))
                dst = buckets.setdefault(key, {"subspace_loss": [], "pc_vs_net_err": [], "eval_relerr": []})
                dst["subspace_loss"].append(loss)
                dst["pc_vs_net_err"].append(pc_vs_net_err)
                dst["eval_relerr"].append(rel_err)

    out: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {p: {} for p in precisions}
    for (p, n), values in buckets.items():
        if not values:
            continue
        out[p][n] = {
            "subspace_loss": _summarize(values.get("subspace_loss", [])),
            "pc_vs_net_err": _summarize(values.get("pc_vs_net_err", [])),
            "eval_relerr": _summarize(values.get("eval_relerr", [])),
        }

    for p in list(out.keys()):
        out[p] = dict(sorted(out[p].items(), key=lambda kv: kv[0]))

    return out


def _print_accuracy_table(
    ns: np.ndarray,
    fp32: Mapping[int, Dict[str, Dict[str, float]]],
    fp16: Mapping[int, Dict[str, Dict[str, float]]],
    table_format: str,
) -> None:
    metrics = ["subspace_loss", "pc_vs_net_err", "eval_relerr"]
    stats = ["mean", "std", "median", "q25", "q75"]

    def _cell(summary: Mapping[str, float], key: str) -> str:
        val = float(summary.get(key, float("nan")))
        if not np.isfinite(float(val)):
            return "nan"
        return f"{float(val):.6g}"

    def _cell_pretty(summary: Mapping[str, float], key: str) -> str:
        val = float(summary.get(key, float("nan")))
        if not np.isfinite(float(val)):
            return "-"
        return f"{float(val):.6g}"

    def _print_grid(headers: List[str], rows: List[List[str]]) -> None:
        widths = [len(h) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
        print(fmt.format(*headers))
        print(fmt.format(*["-" * w for w in widths]))
        for r in rows:
            print(fmt.format(*r))

    def _print_metric_block(metric: str, label: str) -> None:
        print(f"\n[accuracy-table] {label}")
        headers = [
            "N",
            "f32_av",
            "f32_sd",
            "f32_md",
            "f32_q1",
            "f32_q3",
            "f16_av",
            "f16_sd",
            "f16_md",
            "f16_q1",
            "f16_q3",
        ]
        rows: List[List[str]] = []
        for n in ns.astype(np.int64).tolist():
            s32 = fp32.get(int(n)) or {}
            s16 = fp16.get(int(n)) or {}
            if not s32 and not s16:
                continue
            a32 = s32.get(metric, {})
            a16 = s16.get(metric, {})
            rows.append(
                [
                    str(int(n)),
                    _cell_pretty(a32, "mean"),
                    _cell_pretty(a32, "std"),
                    _cell_pretty(a32, "median"),
                    _cell_pretty(a32, "q25"),
                    _cell_pretty(a32, "q75"),
                    _cell_pretty(a16, "mean"),
                    _cell_pretty(a16, "std"),
                    _cell_pretty(a16, "median"),
                    _cell_pretty(a16, "q25"),
                    _cell_pretty(a16, "q75"),
                ]
            )
        _print_grid(headers, rows)

    if str(table_format).lower() == "csv":
        print("\n[accuracy-table]")
        cols = []
        for p in ["fp32", "fp16"]:
            for m in metrics:
                for s in stats:
                    cols.append(f"{p}_{m}_{s}")
        print("N," + ",".join(cols))

        for n in ns.astype(np.int64).tolist():
            s32 = fp32.get(int(n)) or {}
            s16 = fp16.get(int(n)) or {}

            if not s32 and not s16:
                continue

            row: List[str] = []
            for by_p in [s32, s16]:
                for m in metrics:
                    summary = by_p.get(m, {})
                    for st in stats:
                        row.append(_cell(summary, st))

            print(f"{int(n)}," + ",".join(row))
        return

    _print_metric_block("subspace_loss", "subspace_loss")
    _print_metric_block("pc_vs_net_err", "pointcloud_vs_network_error (1-mean)")
    _print_metric_block("eval_relerr", "eval_relerr")


def _align_summary_series(
    ns: np.ndarray,
    stats: Mapping[int, Dict[str, Dict[str, float]]],
    metric: str,
    stat_key: str,
) -> np.ndarray:
    out = np.full((int(ns.size),), float("nan"), dtype=np.float64)
    for i, n in enumerate(ns.astype(np.int64).tolist()):
        per_n = stats.get(int(n)) or {}
        summary = per_n.get(metric) or {}
        out[i] = float(summary.get(stat_key, float("nan")))
    return out


def plot_accuracy_figure(
    fp32: Mapping[int, Dict[str, Dict[str, float]]],
    fp16: Mapping[int, Dict[str, Dict[str, float]]],
    out_dir: Path,
    out_prefix: str,
    formats: Sequence[str],
    dpi: int,
    fig_w: float,
    fig_h: float,
    cfg: StyleConfig,
    legend_ncol: int,
    logx: bool,
    logy: bool,
) -> None:
    if not fp32 and not fp16:
        print("Warning: no data found for fp32/fp16.")
        return

    ns_all = sorted(set(fp32.keys()) | set(fp16.keys()))
    if not ns_all:
        print("Warning: no N found for fp32/fp16.")
        return
    ns = np.asarray(ns_all, dtype=np.float64)

    c_fp32 = "#377EB8"
    c_fp16 = "#4DAF4A"

    def _plot_panel(ax: plt.Axes, metric: str, ylabel: str) -> None:
        y32_med = _align_summary_series(ns, fp32, metric, "median")
        y32_q25 = _align_summary_series(ns, fp32, metric, "q25")
        y32_q75 = _align_summary_series(ns, fp32, metric, "q75")

        y16_med = _align_summary_series(ns, fp16, metric, "median")
        y16_q25 = _align_summary_series(ns, fp16, metric, "q25")
        y16_q75 = _align_summary_series(ns, fp16, metric, "q75")

        if logy:
            eps = 1e-12
            y32_med = np.where(np.isfinite(y32_med), np.maximum(y32_med, eps), y32_med)
            y32_q25 = np.where(np.isfinite(y32_q25), np.maximum(y32_q25, eps), y32_q25)
            y32_q75 = np.where(np.isfinite(y32_q75), np.maximum(y32_q75, eps), y32_q75)
            y16_med = np.where(np.isfinite(y16_med), np.maximum(y16_med, eps), y16_med)
            y16_q25 = np.where(np.isfinite(y16_q25), np.maximum(y16_q25, eps), y16_q25)
            y16_q75 = np.where(np.isfinite(y16_q75), np.maximum(y16_q75, eps), y16_q75)

        ax.plot(ns, y32_med, "-", color=c_fp32, marker="o", label="FP32 (median)", markersize=cfg.marker_size)
        ax.fill_between(ns, y32_q25, y32_q75, color=c_fp32, alpha=0.14, linewidth=0.0)

        ax.plot(ns, y16_med, "-", color=c_fp16, marker="^", label="FP16 (median)", markersize=cfg.marker_size)
        ax.fill_between(ns, y16_q25, y16_q75, color=c_fp16, alpha=0.14, linewidth=0.0)

        if logx:
            ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Number of points ($N$)")
        ax.set_ylabel(ylabel)

        def _fmt(x: float) -> str:
            if x >= 1024**2:
                val = x / (1024**2)
                return f"{val:.0f}M" if val.is_integer() else f"{val:.1f}M"
            if x >= 1024:
                val = x / 1024
                return f"{val:.0f}k" if val.is_integer() else f"{val:.1f}k"
            return str(int(x))

        ax.set_xticks(ns)
        ax.set_xticklabels([_fmt(x) for x in ns])
        ax.set_xlim(float(np.min(ns)), float(np.max(ns)))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.grid(True, which="major")

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.0, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    _plot_panel(ax0, "subspace_loss", r"$\overline{\mathcal{E}}_{\mathrm{sub}}$")
    _plot_panel(ax1, "pc_vs_net_err", r"$\overline{\mathcal{E}}_{\mathrm{vec}}$")
    _plot_panel(ax2, "eval_relerr", r"$\overline{\mathcal{E}}_{\mathrm{val}}$")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=int(legend_ncol),
        fontsize=cfg.legend_fontsize,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        out_path = out_dir / f"{out_prefix}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect accuracy statistics.")
    parser.add_argument("--data_glob", type=str, default="ldata/time_test/*/*/inferred/results.json")
    parser.add_argument("--table_format", type=str, default="pretty", choices=["pretty", "csv"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out_dir", type=str, default="scripts/stats/plots")
    parser.add_argument("--out_prefix", type=str, default="accuracy")
    parser.add_argument("--formats", type=str, default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=11.5)
    parser.add_argument("--height", type=float, default=3.2)

    parser.add_argument("--font", type=str, default="Linux Libertine")
    parser.add_argument("--font_size", type=float, default=10.0)
    parser.add_argument("--axes_labelsize", type=float, default=10.0)
    parser.add_argument("--tick_labelsize", type=float, default=9.0)
    parser.add_argument("--legend_fontsize", type=float, default=9.0)
    parser.add_argument("--title_size", type=float, default=10.0)
    parser.add_argument("--line_width", type=float, default=1.5)
    parser.add_argument("--marker_size", type=float, default=5.0)
    parser.add_argument("--legend_ncol", type=int, default=2)
    parser.add_argument("--logx", dest="logx", action="store_true")
    parser.add_argument("--linearx", dest="logx", action="store_false")
    parser.set_defaults(logx=True)
    parser.add_argument("--logy", dest="logy", action="store_true")
    parser.add_argument("--lineary", dest="logy", action="store_false")
    parser.set_defaults(logy=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    
    data = load_aggregated_stats(
        project_root=project_root,
        pattern=str(args.data_glob),
        precisions=["fp32", "fp16"]
    )

    if not data:
        return

    fp32 = data.get("fp32", {})
    fp16 = data.get("fp16", {})

    ns_all = sorted(set(fp32.keys()) | set(fp16.keys()))
    if not ns_all:
        print("Warning: no N found for fp32/fp16.")
        return
    ns = np.asarray(ns_all, dtype=np.float64)

    _print_accuracy_table(ns, fp32, fp16, table_format=str(args.table_format))

    if bool(args.plot):
        cfg = StyleConfig(
            font=str(args.font),
            font_size=float(args.font_size),
            axes_labelsize=float(args.axes_labelsize),
            tick_labelsize=float(args.tick_labelsize),
            legend_fontsize=float(args.legend_fontsize),
            title_size=float(args.title_size),
            line_width=float(args.line_width),
            marker_size=float(args.marker_size),
        )
        setup_style(cfg)
        out_dir = (project_root / str(args.out_dir)).resolve()
        formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
        plot_accuracy_figure(
            fp32=fp32,
            fp16=fp16,
            out_dir=out_dir,
            out_prefix=str(args.out_prefix),
            formats=formats,
            dpi=int(args.dpi),
            fig_w=float(args.width),
            fig_h=float(args.height),
            cfg=cfg,
            legend_ncol=int(args.legend_ncol),
            logx=bool(args.logx),
            logy=bool(args.logy),
        )


if __name__ == "__main__":
    main()
