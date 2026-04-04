"""Usage: (DO NOT REMOVE THIS COMMENT)
python3 scripts/full_stats_efficiency.py --out_dir scripts/stats/plots_test3 --out_prefix runtime_1x2 --breakdown_show_totals --width 7.75 --height 2.75
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

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
            "text.latex.preamble": r"\usepackage{amsfonts}", # enables \mathbb and other AMS macros
            "mathtext.fontset": "cm",  # Computer Modern, closest to standard LaTeX typography
        }
    )


@dataclass
class AggTimeStats:
    n_points: int
    forward: float
    qr: float
    network_gev: float
    pointcloud_gev: float
    same_residual_gev: float

    @property
    def ours_total(self) -> float:
        return float(self.forward + self.qr + self.network_gev)


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_precision_times(obj: Mapping[str, Any]) -> Dict[str, float]:
    times = obj.get("times", {})
    if not isinstance(times, Mapping):
        return {}
    return {k: _as_float(v) for k, v in times.items()}


def _iter_results_files(project_root: Path, pattern: str) -> List[Path]:
    return list(project_root.glob(pattern))


def _mean_or_nan(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def load_aggregated_times(
    project_root: Path,
    pattern: str,
    precisions: Sequence[str],
) -> Dict[str, Dict[int, AggTimeStats]]:
    files = _iter_results_files(project_root, pattern)
    if not files:
        print(f"Warning: no results.json found for pattern: {pattern}")
        return {}

    print(f"Found {len(files)} results.json files for pattern: {pattern}")

    buckets: Dict[Tuple[str, int], List[AggTimeStats]] = {}

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

        times_top = content.get("times", {})
        if not isinstance(times_top, Mapping):
            times_top = {}
        pointcloud_gev = _as_float(times_top.get("pointcloud_gev"))
        same_residual_top = _as_float(times_top.get("same_residual_gev"))

        precisions_obj = content.get("precisions")
        if isinstance(precisions_obj, Mapping):
            for p in precisions:
                pobj = precisions_obj.get(p)
                if not isinstance(pobj, Mapping):
                    continue
                pt = _parse_precision_times(pobj)
                forward = pt.get("forward", _as_float(times_top.get("forward")))
                qr = pt.get("qr", _as_float(times_top.get("qr")))
                network_gev = pt.get("network_gev", _as_float(times_top.get("network_gev")))
                same_residual_gev = pt.get("same_residual_gev", same_residual_top)
                stats = AggTimeStats(
                    n_points=int(n_points),
                    forward=float(forward),
                    qr=float(qr),
                    network_gev=float(network_gev),
                    pointcloud_gev=float(pointcloud_gev),
                    same_residual_gev=float(same_residual_gev),
                )
                buckets.setdefault((p, int(n_points)), []).append(stats)
        else:
            if "fp32" in precisions:
                stats = AggTimeStats(
                    n_points=int(n_points),
                    forward=_as_float(times_top.get("forward")),
                    qr=_as_float(times_top.get("qr")),
                    network_gev=_as_float(times_top.get("network_gev")),
                    pointcloud_gev=float(pointcloud_gev),
                    same_residual_gev=float(same_residual_top),
                )
                buckets.setdefault(("fp32", int(n_points)), []).append(stats)

    out: Dict[str, Dict[int, AggTimeStats]] = {p: {} for p in precisions}
    for (p, n), lst in buckets.items():
        out[p][n] = AggTimeStats(
            n_points=n,
            forward=_mean_or_nan([x.forward for x in lst]),
            qr=_mean_or_nan([x.qr for x in lst]),
            network_gev=_mean_or_nan([x.network_gev for x in lst]),
            pointcloud_gev=_mean_or_nan([x.pointcloud_gev for x in lst]),
            same_residual_gev=_mean_or_nan([x.same_residual_gev for x in lst]),
        )

    for p in list(out.keys()):
        out[p] = dict(sorted(out[p].items(), key=lambda kv: kv[0]))

    return out


def _parse_int_list(s: str) -> List[int]:
    items: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except Exception:
            continue
    return items


def _select_two_ns(ns: Sequence[int], requested: Sequence[int]) -> List[int]:
    if not ns:
        return []
    ns_sorted = sorted(set(int(x) for x in ns))
    if len(ns_sorted) == 1:
        return [ns_sorted[0]]
    if requested:
        picked: List[int] = []
        for r in requested[:2]:
            best = min(ns_sorted, key=lambda x: abs(x - int(r)))
            if best not in picked:
                picked.append(best)
        if len(picked) == 1:
            picked.append(ns_sorted[-1] if ns_sorted[-1] != picked[0] else ns_sorted[0])
        return picked[:2]
    return [ns_sorted[0], ns_sorted[-1]]


@dataclass(frozen=True)
class FitResult:
    name: str
    params: Tuple[float, ...]
    r2: float
    rel_err: float


def _fit_ours_linear(ns: np.ndarray, ts: np.ndarray) -> Optional[FitResult]:
    mask = np.isfinite(ns) & np.isfinite(ts)
    if int(np.sum(mask)) < 2:
        return None
    x = ns[mask].astype(np.float64)
    y = ts[mask].astype(np.float64)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel_err = float(np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), 1e-12)))
    return FitResult(name="linear", params=(float(a), float(b)), r2=float(r2), rel_err=rel_err)


def _fit_powerlaw(ns: np.ndarray, ts: np.ndarray) -> Optional[FitResult]:
    mask = np.isfinite(ns) & np.isfinite(ts) & (ns > 0) & (ts > 0)
    if int(np.sum(mask)) < 2:
        return None
    x = np.log(ns[mask].astype(np.float64))
    y = np.log(ts[mask].astype(np.float64))
    alpha, logc = np.polyfit(x, y, 1)
    yhat = alpha * x + logc
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel_err = float(np.mean(np.abs(np.exp(y) - np.exp(yhat)) / np.maximum(np.exp(y), 1e-12)))
    c = float(np.exp(logc))
    return FitResult(name="power", params=(c, float(alpha)), r2=float(r2), rel_err=rel_err)


def _format_n(n: int) -> str:
    if n >= 1024:
        if n % 1024 == 0:
            return f"{n // 1024}k"
        return f"{n / 1024.0:.1f}k"
    return str(n)


def _align_series(ns: np.ndarray, stats: Mapping[int, AggTimeStats], key: str) -> np.ndarray:
    out = np.full((int(ns.size),), float("nan"), dtype=np.float64)
    for i, n in enumerate(ns.astype(np.int64).tolist()):
        s = stats.get(int(n))
        if s is None:
            continue
        out[i] = float(getattr(s, key))
    return out


def _print_scaling_table_combined(
    ns: np.ndarray,
    fp32: Mapping[int, AggTimeStats],
    fp16: Mapping[int, AggTimeStats],
    same_residual_fallback_to_full: bool,
) -> None:
    print("\n[scaling-table] (seconds)")
    print(
        "N,"
        "fp32_forward,fp32_qr,fp32_network_gev,fp32_ours_total,fp32_arpack_full,fp32_arpack_same,"
        "fp16_forward,fp16_qr,fp16_network_gev,fp16_ours_total,fp16_arpack_full,fp16_arpack_same"
    )
    for n in ns.astype(np.int64).tolist():
        s32 = fp32.get(int(n))
        s16 = fp16.get(int(n))
        if s32 is None and s16 is None:
            continue

        def _cell(s: Optional[AggTimeStats], attr: str) -> float:
            if s is None:
                return float("nan")
            return float(getattr(s, attr))

        s32_same = _cell(s32, "same_residual_gev")
        s16_same = _cell(s16, "same_residual_gev")
        if same_residual_fallback_to_full:
            if not np.isfinite(s32_same) and s32 is not None:
                s32_same = float(s32.pointcloud_gev)
            if not np.isfinite(s16_same) and s16 is not None:
                s16_same = float(s16.pointcloud_gev)

        print(
            f"{int(n)},"
            f"{_cell(s32,'forward'):.6g},{_cell(s32,'qr'):.6g},{_cell(s32,'network_gev'):.6g},{_cell(s32,'ours_total'):.6g},"
            f"{_cell(s32,'pointcloud_gev'):.6g},{s32_same:.6g},"
            f"{_cell(s16,'forward'):.6g},{_cell(s16,'qr'):.6g},{_cell(s16,'network_gev'):.6g},{_cell(s16,'ours_total'):.6g},"
            f"{_cell(s16,'pointcloud_gev'):.6g},{s16_same:.6g}"
        )


def _plot_scaling_panel_merged(
    ax: plt.Axes,
    title: str,
    ns: np.ndarray,
    ours_fp32: np.ndarray,
    ours_fp16: np.ndarray,
    ar_full_fp32: np.ndarray,
    ar_full_fp16: np.ndarray,
    ar_same_fp32: np.ndarray,
    ar_same_fp16: np.ndarray,
    fits_full: Mapping[str, Optional[FitResult]],
    fits_same: Mapping[str, Optional[FitResult]],
    cfg: StyleConfig,
    legend_ncol: int,
) -> None:
    # Colors from Set1 or Tableau
    c_ours_32 = "#377EB8"  # Blue
    c_ours_16 = "#4DAF4A"  # Greenish
    c_arpack = "#B2182B"   # Deep red
    c_arpack_relaxed = "#D6604D"  # Dark salmon red

    # Ours
    ax.loglog(ns, ours_fp32, "-", color=c_ours_32, marker="o", label="Ours (FP32)", markersize=cfg.marker_size)
    ax.loglog(ns, ours_fp16, "-", color=c_ours_16, marker="^", label="Ours (FP16)", markersize=cfg.marker_size)

    # ARPACK Full
    ax.loglog(ns, ar_full_fp32, "-", color=c_arpack, marker="s", label="ARPACK (Accurate)", markersize=cfg.marker_size)

    # ARPACK Same
    ax.loglog(ns, ar_same_fp32, "--", color=c_arpack_relaxed, marker="s", alpha=0.9, label="ARPACK (Relaxed)", markersize=cfg.marker_size)

    # Plot fit lines (faint) and annotate text
    # Function to add annotation
    def _annotate_fit(name: str, fr: Optional[FitResult], x_vals: np.ndarray, y_vals: np.ndarray, color: str) -> None:
        if fr is None:
            return
        
        # Plot the fit line faint
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(mask):
            return
        
        valid_ns = x_vals[mask]
        
        if fr.name == "linear":
            a, b = fr.params
            y_fit = a * valid_ns + b
            label_text = r"$\mathcal{O}(N)$"
        else:
            c, alpha = fr.params
            y_fit = c * (valid_ns ** alpha)
            label_text = rf"$\mathcal{{O}}(N^{{{alpha:.2f}}})$"
            
        ax.loglog(valid_ns, y_fit, ":", color=color, alpha=0.4, linewidth=1.0)

        # Annotate at the end
        idx = -1
        if len(valid_ns) > 0:
            x_end = valid_ns[idx]
            y_end = y_fit[idx]
            # Offset slightly
            ax.text(x_end, y_end * 1.15, label_text, color=color, fontsize=cfg.font_size, ha="left", va="bottom")

    def _add_reference_slope_guide(
        slope: float,
        label_text: str,
        *,
        y_left_target: float,
        below_factor: float,
    ) -> None:
        mask = np.isfinite(ns) & np.isfinite(ours_fp32) & np.isfinite(ours_fp16)
        if not np.any(mask):
            return

        valid_ns = ns[mask]
        valid_fp32 = ours_fp32[mask]
        valid_fp16 = ours_fp16[mask]
        x1 = float(valid_ns[-1])
        y1 = float(min(valid_fp32[-1], valid_fp16[-1]) * below_factor)
        if not (np.isfinite(x1) and np.isfinite(y1) and x1 > 0 and y1 > 0):
            return

        y0 = float(y_left_target)
        if not (np.isfinite(y0) and y0 > 0):
            return

        k = y1 / (x1 ** slope)
        x0 = (y0 / k) ** (1.0 / slope)

        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        if not (np.isfinite(x_lo) and np.isfinite(x_hi) and np.isfinite(y_lo) and np.isfinite(y_hi)):
            return

        x0 = float(np.clip(x0, max(x_lo, 1e-12), x1))
        y0 = float(k * (x0 ** slope))
        if not (np.isfinite(y0) and y0 > 0):
            return

        ax.loglog(
            [x0, x1],
            [y0, y1],
            linestyle=(0, (4, 3)),
            color="0.35",
            alpha=0.65,
            linewidth=1.0,
            solid_capstyle="butt",
        )

        ax.text(
            x1,
            y1 * 0.3,
            label_text,
            color="0.2",
            alpha=0.9,
            fontsize=cfg.font_size,
            ha="left",
            va="bottom",
        )

    _add_reference_slope_guide(1.0, r"$\mathcal{O}(N)$", y_left_target=1e-2, below_factor=0.7)
    
    # Annotate ARPACK Full - shift text up slightly
    if fits_full.get("arpack_fp32"):
        fr = fits_full["arpack_fp32"]
        mask = np.isfinite(ns) & np.isfinite(ar_full_fp32)
        if np.any(mask):
            # Recalculate fit y
            c, alpha = fr.params
            valid_ns = ns[mask]
            y_fit = c * (valid_ns ** alpha)
            # Plot line
            ax.loglog(valid_ns, y_fit, ":", color=c_arpack, alpha=0.4, linewidth=1.0)
            # Text with offset
            ax.text(valid_ns[-1], y_fit[-1] * 1.25, rf"$\mathcal{{O}}(N^{{{alpha:.2f}}})$", color=c_arpack, fontsize=cfg.font_size, ha="left", va="bottom")

    # Annotate ARPACK Same - shift text down slightly
    if fits_same.get("arpack_fp32"):
        fr = fits_same["arpack_fp32"]
        mask = np.isfinite(ns) & np.isfinite(ar_same_fp32)
        if np.any(mask):
             # Recalculate fit y
            c, alpha = fr.params
            valid_ns = ns[mask]
            y_fit = c * (valid_ns ** alpha)
            # Plot line
            ax.loglog(valid_ns, y_fit, ":", color=c_arpack_relaxed, alpha=0.4, linewidth=1.0)
            # Text with offset
            ax.text(valid_ns[-1], y_fit[-1] / 1.15, rf"$\mathcal{{O}}(N^{{{alpha:.2f}}})$", color=c_arpack_relaxed, fontsize=cfg.font_size, ha="left", va="top")
    
    # ax.set_title(title)
    ax.set_xlabel("Number of Points $N$")
    ax.set_ylabel("Time (s)")

    ax.grid(True, which="major", ls="-", alpha=0.6)
    ax.grid(True, which="minor", ls=":", alpha=0.35)
    # Move legend to upper left to utilize empty space
    ax.legend(
        loc="upper left",
        ncol=1,
        fontsize=cfg.legend_fontsize,
        framealpha=0.9,
    )



def _fmt_seconds(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if x < 1.0:
        return f"{x * 1000.0:.0f}ms"
    return f"{x:.2f}s"


def _plot_breakdown_combined(
    ax: plt.Axes,
    n_small: int,
    n_big: int,
    fp32: Mapping[int, AggTimeStats],
    fp16: Mapping[int, AggTimeStats],
    mode: str,
    show_totals: bool,
    cfg: StyleConfig,
    legend_ncol: int,
) -> None:
    stats_small_fp16 = fp16.get(int(n_small))
    stats_small_fp32 = fp32.get(int(n_small))
    stats_big_fp16 = fp16.get(int(n_big))
    stats_big_fp32 = fp32.get(int(n_big))

    # Baselines (fp32 total time)
    base_small = stats_small_fp32.ours_total if stats_small_fp32 else float("nan")
    base_big = stats_big_fp32.ours_total if stats_big_fp32 else float("nan")

    entries: List[Tuple[str, Optional[AggTimeStats], float]] = [
        (f"N={_format_n(int(n_small))} fp16", stats_small_fp16, base_small),
        (f"N={_format_n(int(n_small))} fp32", stats_small_fp32, base_small),
        (f"N={_format_n(int(n_big))} fp16", stats_big_fp16, base_big),
        (f"N={_format_n(int(n_big))} fp32", stats_big_fp32, base_big),
    ]

    xs = np.asarray([0.0, 1.0, 3.0, 4.0], dtype=np.float64)
    width = 0.75

    def _vals(s: Optional[AggTimeStats], baseline: float) -> Tuple[float, float, float, float]:
        if s is None or not np.isfinite(baseline) or baseline <= 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        # Normalize by baseline (fp32 total) -> percentage of fp32 time
        scale = 100.0 / baseline
        f = float(s.forward) * scale
        q = float(s.qr) * scale
        r = float(s.network_gev) * scale
        t = float(s.ours_total) * scale
        return f, q, r, t

    f = np.asarray([_vals(s, b)[0] for _, s, b in entries], dtype=np.float64)
    q = np.asarray([_vals(s, b)[1] for _, s, b in entries], dtype=np.float64)
    r = np.asarray([_vals(s, b)[2] for _, s, b in entries], dtype=np.float64)
    t = np.asarray([_vals(s, b)[3] for _, s, b in entries], dtype=np.float64)

    # Colors from Set1
    c_backbone = "#377EB8"  # Blue
    c_qr = "#FF7F00"        # Orange
    c_eig = "#984EA3"       # Purple

    # Plot stacked bars
    ax.bar(xs, f, width=width, color=c_backbone, alpha=0.9, label="Forward", edgecolor="white", linewidth=0.5)
    ax.bar(xs, q, width=width, bottom=f, color=c_qr, alpha=0.9, label="Weighted QR", edgecolor="white", linewidth=0.5)
    ax.bar(xs, r, width=width, bottom=f+q, color=c_eig, alpha=0.9, label="Rayleigh-Ritz", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Relative Runtime")
    # Set ylim to accommodate labels (e.g., 125%)
    ax.set_ylim(0.0, 110)
    
    # Add percentage formatter to Y axis
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    ax.grid(True, axis="y", which="major", ls="-", alpha=0.4)

    ax.set_xlim(-0.8, 4.8)
    
    # X-axis labels: primary labels (fp16, fp32) at the ticks
    ax.set_xticks(xs)
    ax.set_xticklabels(["FP16", "FP32", "FP16", "FP32"], fontsize=cfg.tick_labelsize)
    
    # Add a common X-axis label to match the left plot for alignment
    # ax.set_xlabel("Number of Points $N$")

    # Secondary labels (N=...) below the groups, above the X-label
    # Positioning using data coordinates for X and axes coordinates for Y
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.5, -0.16, f"$N={_format_n(int(n_small))}$", transform=trans, 
            ha="center", va="top", fontsize=cfg.axes_labelsize, fontweight="bold")
    ax.text(3.5, -0.16, f"$N={_format_n(int(n_big))}$", transform=trans, 
            ha="center", va="top", fontsize=cfg.axes_labelsize, fontweight="bold")

    # Remove the old twiny axis logic and just use the main ax for legend
    if show_totals:
        for i, (xi, total_pct) in enumerate(zip(xs.tolist(), t.tolist())):
            if not np.isfinite(total_pct):
                continue
            # Retrieve original total time for label
            stat = entries[i][1]
            if stat:
                orig_total = stat.ours_total
                txt = _fmt_seconds(orig_total)
                # Position text above bar, bolded
                ax.text(xi, total_pct + 3.0, txt, ha="center", va="bottom", 
                        fontsize=cfg.font_size * 0.9, fontweight="bold")

    # Legend inside the plot area, positioned above the N=big fp16 bar (around x=3.0)
    # Using bbox_to_anchor to place it precisely in the middle-right empty space
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.385, 1.03),
        ncol=1,
        fontsize=cfg.legend_fontsize,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )




def plot_combined_figure(
    fp32: Dict[int, AggTimeStats],
    fp16: Dict[int, AggTimeStats],
    out_dir: Path,
    out_prefix: str,
    formats: Sequence[str],
    dpi: int,
    fig_w: float,
    fig_h: float,
    breakdown_ns: Sequence[int],
    breakdown_mode: str,
    breakdown_show_totals: bool,
    cfg: StyleConfig,
    legend_ncol: int,
) -> None:
    if not fp32 and not fp16:
        print("Warning: no data found for fp32/fp16.")
        return

    ns_all = sorted(set(fp32.keys()) | set(fp16.keys()))
    if not ns_all:
        print("Warning: no N found for fp32/fp16.")
        return
    ns = np.asarray(ns_all, dtype=np.float64)

    ours_fp32 = _align_series(ns, fp32, "ours_total")
    ours_fp16 = _align_series(ns, fp16, "ours_total")
    ar_full_fp32 = _align_series(ns, fp32, "pointcloud_gev")
    ar_full_fp16 = _align_series(ns, fp16, "pointcloud_gev")
    ar_same_fp32 = _align_series(ns, fp32, "same_residual_gev")
    ar_same_fp16 = _align_series(ns, fp16, "same_residual_gev")

    same_residual_fallback = False
    if not np.all(np.isfinite(ar_same_fp32)) or not np.all(np.isfinite(ar_same_fp16)):
        print("Warning: missing same_residual_gev for some (precision,N); using pointcloud_gev as fallback.")
        same_residual_fallback = True
        ar_same_fp32 = np.where(np.isfinite(ar_same_fp32), ar_same_fp32, ar_full_fp32)
        ar_same_fp16 = np.where(np.isfinite(ar_same_fp16), ar_same_fp16, ar_full_fp16)

    _print_scaling_table_combined(ns=ns, fp32=fp32, fp16=fp16, same_residual_fallback_to_full=same_residual_fallback)

    def _print_fit_block(tag: str, ours32: np.ndarray, ours16: np.ndarray, ar32: np.ndarray, ar16: np.ndarray) -> Mapping[str, Optional[FitResult]]:
        fits: Dict[str, Optional[FitResult]] = {}
        fits["ours_fp32"] = _fit_ours_linear(ns, ours32)
        fits["ours_fp16"] = _fit_ours_linear(ns, ours16)
        fits["arpack_fp32"] = _fit_powerlaw(ns, ar32)
        fits["arpack_fp16"] = _fit_powerlaw(ns, ar16)

        print(f"\n[scaling-fit] setting={tag}")
        for name, fr in [
            ("Ours(fp32)", fits["ours_fp32"]),
            ("Ours(fp16)", fits["ours_fp16"]),
            ("ARPACK", fits["arpack_fp32"]),
        ]:
            if fr is None:
                print(f"{name}: insufficient points for fit")
                continue
            if fr.name == "linear":
                a, b = fr.params
                print(f"{name}: t(N)≈aN+b  a={a:.6g} b={b:.6g}  R2={fr.r2:.4f}  rel_err={fr.rel_err:.4f}")
            else:
                c, alpha = fr.params
                print(f"{name}: t(N)≈cN^α  c={c:.6g} α={alpha:.4f}  R2={fr.r2:.4f}  rel_err={fr.rel_err:.4f}")
        return fits

    fits_full = _print_fit_block("full", ours_fp32, ours_fp16, ar_full_fp32, ar_full_fp16)
    fits_same = _print_fit_block("same_residual", ours_fp32, ours_fp16, ar_same_fp32, ar_same_fp16)

    ns_breakdown_candidates = sorted(set(fp32.keys()) & set(fp16.keys()))
    if not ns_breakdown_candidates:
        ns_breakdown_candidates = ns_all
    picked = _select_two_ns(ns_breakdown_candidates, list(breakdown_ns))
    if len(picked) == 1:
        picked = [picked[0], picked[0]]
    n_small, n_big = int(picked[0]), int(picked[1])

    print("\n[breakdown-table] (seconds)")
    print("N,precision,forward,qr,network_gev,ours_total")
    for n in [n_small, n_big]:
        for p_name, d in [("fp16", fp16), ("fp32", fp32)]:
            s = d.get(int(n))
            if s is None:
                print(f"{int(n)},{p_name},nan,nan,nan,nan")
            else:
                print(f"{int(n)},{p_name},{s.forward:.6g},{s.qr:.6g},{s.network_gev:.6g},{s.ours_total:.6g}")

    # Layout: 1x2 grid
    # Left: Scaling (Merged)
    # Right: Breakdown
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.0])
    ax_scaling = fig.add_subplot(gs[0, 0])
    ax_breakdown = fig.add_subplot(gs[0, 1])

    _plot_scaling_panel_merged(
        ax_scaling,
        title="Scaling Comparison",
        ns=ns,
        ours_fp32=ours_fp32,
        ours_fp16=ours_fp16,
        ar_full_fp32=ar_full_fp32,
        ar_full_fp16=ar_full_fp16,
        ar_same_fp32=ar_same_fp32,
        ar_same_fp16=ar_same_fp16,
        fits_full=fits_full,
        fits_same=fits_same,
        cfg=cfg,
        legend_ncol=legend_ncol,
    )

    _plot_breakdown_combined(
        ax_breakdown,
        n_small=n_small,
        n_big=n_big,
        fp32=fp32,
        fp16=fp16,
        mode=breakdown_mode,
        show_totals=bool(breakdown_show_totals),
        cfg=cfg,
        legend_ncol=legend_ncol,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="scripts/stats/plots")
    p.add_argument("--out_prefix", type=str, default="runtime")
    p.add_argument("--data_glob", type=str, default="ldata/time_test/*/*/inferred/results.json")
    p.add_argument("--formats", type=str, default="pdf,png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--ignore", type=int, nargs="+", default=[])

    p.add_argument("--width", type=float, default=16.0)  # Increased width for 1x2 layout
    p.add_argument("--height", type=float, default=6.0)

    p.add_argument("--font", type=str, default="Linux Libertine")
    p.add_argument("--font_size", type=float, default=10.0)
    p.add_argument("--axes_labelsize", type=float, default=10.0)
    p.add_argument("--tick_labelsize", type=float, default=9.0)
    p.add_argument("--legend_fontsize", type=float, default=9.0)
    p.add_argument("--title_size", type=float, default=10.0)
    p.add_argument("--line_width", type=float, default=1.5)
    p.add_argument("--marker_size", type=float, default=5.0)

    p.add_argument("--legend_ncol", type=int, default=2)

    p.add_argument("--breakdown_ns", type=str, default="32768,524288")
    p.add_argument("--breakdown_mode", type=str, default="percent")
    p.add_argument("--breakdown_show_totals", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    out_dir = (project_root / args.out_dir).resolve()

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

    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    breakdown_ns = _parse_int_list(str(args.breakdown_ns))

    data = load_aggregated_times(project_root=project_root, pattern=str(args.data_glob), precisions=["fp32", "fp16"])
    if not data:
        return
    ignore_set = set(int(x) for x in (args.ignore or []))
    if ignore_set:
        for p in ["fp32", "fp16"]:
            d = data.get(p)
            if not d:
                continue
            data[p] = {int(k): v for k, v in d.items() if int(k) not in ignore_set}

    plot_combined_figure(
        fp32=data.get("fp32", {}),
        fp16=data.get("fp16", {}),
        out_dir=out_dir,
        out_prefix=str(args.out_prefix),
        formats=formats,
        dpi=int(args.dpi),
        fig_w=float(args.width),
        fig_h=float(args.height),
        breakdown_ns=breakdown_ns,
        breakdown_mode=str(args.breakdown_mode),
        breakdown_show_totals=bool(args.breakdown_show_totals),
        cfg=cfg,
        legend_ncol=int(args.legend_ncol),
    )



if __name__ == "__main__":
    main()
