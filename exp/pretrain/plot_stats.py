from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _align_length(x: np.ndarray, k: int, fill_value: float) -> np.ndarray:
    x = _as_float_array(x).reshape(-1)
    if x.size == k:
        return x
    if x.size > k:
        return x[:k]
    out = np.full((k,), fill_value, dtype=np.float64)
    out[: x.size] = x
    return out


@dataclass(frozen=True)
class ValidationCase:
    n_points: int
    scores: np.ndarray
    eval_relerr: np.ndarray
    subloss: float
    per_mode_loss: np.ndarray
    eval_time: float


def _parse_case(d: dict[str, Any], k: int | None) -> ValidationCase | None:
    n_points = d.get("n_points")
    if not isinstance(n_points, int) or n_points <= 0:
        return None

    scores = d.get("scores", {}).get("pointcloud_vs_network")
    if scores is None:
        return None
    scores_arr = _as_float_array(scores).reshape(-1)

    if k is None:
        k_picked = int(scores_arr.size)
    else:
        k_picked = int(k)

    scores_arr = _align_length(scores_arr, k_picked, fill_value=float("nan"))

    relerr = d.get("eval_relerr", [])
    relerr_arr = _as_float_array(relerr).reshape(-1)
    if relerr_arr.size == k_picked - 1:
        relerr_arr = np.concatenate([np.asarray([float("nan")], dtype=np.float64), relerr_arr], axis=0)
    relerr_arr = _align_length(relerr_arr, k_picked, fill_value=float("nan"))

    subloss = d.get("subloss")
    try:
        subloss_f = float(subloss)
    except Exception:
        subloss_f = float("nan")

    eval_time = d.get("eval_time")
    try:
        eval_time_f = float(eval_time)
    except Exception:
        eval_time_f = float("nan")

    per_mode_loss_raw = d.get("per_mode_loss")
    if per_mode_loss_raw is None:
        per_mode_loss_arr = np.full((k_picked,), float("nan"), dtype=np.float64)
    else:
        per_mode_loss_arr = np.asarray(per_mode_loss_raw, dtype=np.float64)
        if per_mode_loss_arr.ndim == 2:
            per_mode_energy = np.nansum(per_mode_loss_arr, axis=0)
            per_mode_loss_arr = 1.0 - per_mode_energy
        per_mode_loss_arr = _align_length(per_mode_loss_arr.reshape(-1), k_picked, fill_value=float("nan"))

    return ValidationCase(
        n_points=int(n_points),
        scores=scores_arr,
        eval_relerr=relerr_arr,
        subloss=subloss_f,
        per_mode_loss=per_mode_loss_arr,
        eval_time=eval_time_f,
    )


def _load_validation_json(path: Path, k: int | None) -> list[ValidationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {path}, got {type(raw)}")

    cases: list[ValidationCase] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        c = _parse_case(x, k=k)
        if c is not None:
            cases.append(c)
    return cases


def _percentiles(x: np.ndarray, ps: list[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed=42)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    samples = x[indices]
    means = np.mean(samples, axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    return lower, upper


def _compute_dist_stats(x: np.ndarray, ps: list[float] | None = None, max_percentile: float = 100.0) -> dict[str, float]:
    if ps is None:
        ps = [25, 50, 75, 90, 99]
    stats = _percentiles(x, ps)
    
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    
    if max_percentile < 100.0 and x.size > 0:
        limit = np.percentile(x, max_percentile)
        x = x[x <= limit]

    if x.size > 0:
        stats["mean"] = float(np.mean(x))
        stats["std"] = float(np.std(x))
        stats["count"] = float(x.size)
        lo, hi = _bootstrap_ci(x)
        stats["mean_ci_lower"] = lo
        stats["mean_ci_upper"] = hi
    else:
        stats["mean"] = float("nan")
        stats["std"] = float("nan")
        stats["count"] = 0.0
        stats["mean_ci_lower"] = float("nan")
        stats["mean_ci_upper"] = float("nan")
    return stats


def _group_by_n_points(cases: list[ValidationCase]) -> dict[int, list[ValidationCase]]:
    groups: dict[int, list[ValidationCase]] = {}
    for c in cases:
        groups.setdefault(int(c.n_points), []).append(c)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def _compute_group_stats(group: list[ValidationCase], max_percentile: float = 100.0) -> dict[str, Any]:
    if not group:
        return {"case_count": 0}

    k = int(group[0].scores.size)
    scores = np.stack([c.scores for c in group], axis=0)
    scores = np.clip(scores, -1.0, 1.0)
    mse = 1.0 - scores

    relerr = np.stack([c.eval_relerr for c in group], axis=0)
    subloss = np.asarray([c.subloss for c in group], dtype=np.float64)
    per_mode_loss = np.stack([c.per_mode_loss for c in group], axis=0)
    eval_time = np.asarray([c.eval_time for c in group], dtype=np.float64)

    per_case_median_mse = np.median(mse, axis=1)
    mse_k = mse[:, k - 1]
    relerr_k = relerr[:, k - 1]
    relerr_ex0 = relerr[:, 1:] if k >= 2 else relerr[:, :0]
    relerr_ex0_flat = relerr_ex0.reshape(-1)

    return {
        "case_count": int(len(group)),
        "k": int(k),
        "median_per_mode_loss_by_mode": np.nanmedian(per_mode_loss, axis=0).tolist(),
        "p25_per_mode_loss_by_mode": np.nanpercentile(per_mode_loss, 25, axis=0).tolist(),
        "p75_per_mode_loss_by_mode": np.nanpercentile(per_mode_loss, 75, axis=0).tolist(),
        "median_score_by_mode": np.median(scores, axis=0).tolist(),
        "median_mse_by_mode": np.median(mse, axis=0).tolist(),
        "p25_mse_by_mode": np.percentile(mse, 25, axis=0).tolist(),
        "p75_mse_by_mode": np.percentile(mse, 75, axis=0).tolist(),
        "median_eval_relerr_by_mode_excl0": np.nanmedian(relerr_ex0, axis=0).tolist() if k >= 2 else [],
        "eval_relerr_all_modes_excl0": {
            **_percentiles(relerr_ex0_flat, [10, 50, 90, 95, 99]),
            "mean": float(np.mean(relerr_ex0_flat[np.isfinite(relerr_ex0_flat)]))
            if np.any(np.isfinite(relerr_ex0_flat))
            else float("nan"),
            "std": float(np.std(relerr_ex0_flat[np.isfinite(relerr_ex0_flat)]))
            if np.any(np.isfinite(relerr_ex0_flat))
            else float("nan"),
        },
        "per_case_median_mse": _compute_dist_stats(per_case_median_mse, max_percentile=max_percentile),
        "mse_at_mode_k": _compute_dist_stats(mse_k, max_percentile=max_percentile),
        "eval_relerr_at_mode_k": _compute_dist_stats(relerr_k, max_percentile=max_percentile),
        "subloss": _compute_dist_stats(subloss, max_percentile=max_percentile),
        "eval_time": _compute_dist_stats(eval_time, max_percentile=max_percentile),
    }


def _plot_summary(
    groups: dict[int, list[ValidationCase]],
    groups_fp16: dict[int, list[ValidationCase]] | None,
    out_path: Path,
    cfg: StyleConfig,
    title: str | None,
    show_iqr: bool,
    logy_mse: bool,
    logy_subloss: bool,
    figsize: tuple[float, float],
) -> None:
    if not groups:
        raise ValueError("No groups to plot.")

    setup_style(cfg)

    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]},
        constrained_layout=True,
    )
    colors = plt.cm.tab10.colors

    relerr_vals_all: list[np.ndarray] = []
    for i, (n_points, cases) in enumerate(groups.items()):
        stats = _compute_group_stats(cases)
        k = int(stats["k"])
        x = np.arange(1, k + 1)
        pml_med = np.asarray(stats["median_per_mode_loss_by_mode"], dtype=np.float64)
        pml_p25 = np.asarray(stats["p25_per_mode_loss_by_mode"], dtype=np.float64)
        pml_p75 = np.asarray(stats["p75_per_mode_loss_by_mode"], dtype=np.float64)
        mse_med = np.asarray(stats["median_mse_by_mode"], dtype=np.float64)
        mse_p25 = np.asarray(stats["p25_mse_by_mode"], dtype=np.float64)
        mse_p75 = np.asarray(stats["p75_mse_by_mode"], dtype=np.float64)

        color = colors[i % len(colors)]
        label_fp32 = f"N={n_points}"

        if logy_mse:
            mse_med = np.maximum(mse_med, 1e-15)
            mse_p25 = np.maximum(mse_p25, 1e-15)
            mse_p75 = np.maximum(mse_p75, 1e-15)

        if logy_subloss:
            pml_med = np.maximum(pml_med, 1e-15)
            pml_p25 = np.maximum(pml_p25, 1e-15)
            pml_p75 = np.maximum(pml_p75, 1e-15)

        ax0.plot(x, pml_med, color=color, label=label_fp32)
        if show_iqr:
            ax0.fill_between(x, pml_p25, pml_p75, color=color, alpha=0.14, linewidth=0.0)

        ax1.plot(x, mse_med, color=color, label=label_fp32)
        if show_iqr:
            ax1.fill_between(x, mse_p25, mse_p75, color=color, alpha=0.14, linewidth=0.0)

        relerr_k = np.asarray([c.eval_relerr[k - 1] for c in cases], dtype=np.float64)
        relerr_k = relerr_k[np.isfinite(relerr_k)]
        if relerr_k.size:
            relerr_vals_all.append(relerr_k)

        if groups_fp16 is not None and int(n_points) in groups_fp16:
            stats16 = _compute_group_stats(groups_fp16[int(n_points)])
            k16 = int(stats16["k"])
            x16 = np.arange(1, k16 + 1)
            pml16 = np.asarray(stats16["median_per_mode_loss_by_mode"], dtype=np.float64)
            mse16 = np.asarray(stats16["median_mse_by_mode"], dtype=np.float64)
            if logy_mse:
                mse16 = np.maximum(mse16, 1e-15)
            if logy_subloss:
                pml16 = np.maximum(pml16, 1e-15)
            label_fp16 = f"N={n_points} fp16 (n={stats16['case_count']})"
            ax0.plot(x16, pml16, color=color, linestyle="--", alpha=0.95, label=label_fp16)
            ax1.plot(x16, mse16, color=color, linestyle="--", alpha=0.95, label=label_fp16)

            relerr16_k = np.asarray([c.eval_relerr[k16 - 1] for c in groups_fp16[int(n_points)]], dtype=np.float64)
            relerr16_k = relerr16_k[np.isfinite(relerr16_k)]
            if relerr16_k.size:
                relerr_vals_all.append(relerr16_k)

    ax0.set_ylabel(r"Median SpanLoss $\mathcal{E}_{\mathrm{span}}(i)$")
    ax0.set_xlabel("Mode $i$")
    if logy_subloss:
        ax0.set_yscale("log")
        ax0.set_ylim(bottom=1e-5)
    else:
        ax0.set_ylim(bottom=0.0)

    ax1.set_ylabel(r"Median MSE $\mathcal{E}_{\mathrm{vec}}(i)$")
    ax1.set_xlabel("Mode $i$")
    if logy_mse:
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=1e-5)
    else:
        ax1.set_ylim(bottom=0.0)

    for ax in (ax0, ax1):
        ax.set_xlim(0, 96)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.grid(True, which="major")

    relerr_all = np.concatenate(relerr_vals_all, axis=0) if relerr_vals_all else np.asarray([], dtype=np.float64)
    relerr_all = relerr_all[np.isfinite(relerr_all)]
    if relerr_all.size:
        hi = float(np.quantile(relerr_all, 0.995))
        hi = max(hi, 1e-6)
    else:
        hi = 1.0
    bins = np.linspace(0.0, hi * 1.05, 41)

    for i, (n_points, cases) in enumerate(groups.items()):
        k = int(_compute_group_stats(cases)["k"])
        relerr_k = np.asarray([c.eval_relerr[k - 1] for c in cases], dtype=np.float64)
        relerr_k = relerr_k[np.isfinite(relerr_k)]
        if relerr_k.size:
            h, e = np.histogram(relerr_k, bins=bins, density=True)
            c = 0.5 * (e[:-1] + e[1:])
            color = colors[i % len(colors)]
            label = f"N={n_points}"
            ax2.plot(
                c,
                h,
                color=color,
                linewidth=cfg.line_width,
                label=label,
            )
            ax2.fill_between(c, h, color=color, alpha=0.15)

        if groups_fp16 is not None and int(n_points) in groups_fp16:
            cases16 = groups_fp16[int(n_points)]
            k16 = int(_compute_group_stats(cases16)["k"])
            relerr16_k = np.asarray([c.eval_relerr[k16 - 1] for c in cases16], dtype=np.float64)
            relerr16_k = relerr16_k[np.isfinite(relerr16_k)]
            if relerr16_k.size:
                h16, e16 = np.histogram(relerr16_k, bins=bins, density=True)
                c16 = 0.5 * (e16[:-1] + e16[1:])
                color = colors[i % len(colors)]
                label = f"N={n_points} fp16 (n={len(relerr16_k)})"
                ax2.plot(
                    c16,
                    h16,
                    color=color,
                    linewidth=cfg.line_width,
                    linestyle="--",
                    label=label,
                )
                ax2.fill_between(c16, h16, color=color, alpha=0.08)

    ax2.set_xlabel(r"Rel. eigenvalue error $\mathcal{E}_{\mathrm{val}}(i)$")
    ax2.set_ylabel("Density %")
    ax2.set_xlim(0.0, hi * 1.05)
    ax2.set_ylim(bottom=0.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(top=False, right=False)
    ax2.grid(True, which="major")

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label: dict[str, Any] = {}
    for handle, label in list(zip(handles0, labels0)) + list(zip(handles1, labels1)) + list(zip(handles2, labels2)):
        if label and label not in by_label:
            by_label[label] = handle
    # if by_label:
    #     ncol = min(len(by_label), 4)
    #     fig.legend(
    #         list(by_label.values()),
    #         list(by_label.keys()),
    #         loc="lower center",
    #         bbox_to_anchor=(0.5, 1.02),
    #         ncol=ncol,
    #         frameon=False,
    #     )

    if title:
        fig.suptitle(title, y=1.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="outputs.json")
    p.add_argument("--input_fp16", type=str, default="")
    p.add_argument("--out_dir", type=str, default="validation_plots")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--title", type=str, default="")
    p.add_argument("--no_group_by_n_points", action="store_true")
    p.add_argument("--show_iqr", action="store_true")
    p.add_argument("--logy_mse", action="store_true")
    p.add_argument("--logy_subloss", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--width", type=float, default=12.6)
    p.add_argument("--height", type=float, default=4.0)
    p.add_argument("--percentile", type=float, default=100.0, help="Percentile threshold to filter outliers for stats (0-100)")

    p.add_argument("--font", type=str, default="Linux Libertine")
    p.add_argument("--font_size", type=float, default=10.0)
    p.add_argument("--axes_labelsize", type=float, default=10.0)
    p.add_argument("--tick_labelsize", type=float, default=9.0)
    p.add_argument("--legend_fontsize", type=float, default=9.0)
    p.add_argument("--title_size", type=float, default=10.0)
    p.add_argument("--line_width", type=float, default=1.5)
    p.add_argument("--marker_size", type=float, default=5.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    input_fp16_path = Path(args.input_fp16) if args.input_fp16.strip() else None
    if input_fp16_path is not None and not input_fp16_path.is_absolute():
        input_fp16_path = (Path.cwd() / input_fp16_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    k = int(args.k) if int(args.k) > 0 else None
    cases = _load_validation_json(input_path, k=k)
    if not cases:
        raise ValueError(f"No valid cases found in {input_path}")

    cases_fp16: list[ValidationCase] | None = None
    if input_fp16_path is not None and input_fp16_path.exists():
        cases_fp16 = _load_validation_json(input_fp16_path, k=k)

    if args.no_group_by_n_points:
        groups = {int(cases[0].n_points): cases}
    else:
        groups = _group_by_n_points(cases)

    groups_fp16 = None
    if cases_fp16:
        if args.no_group_by_n_points:
            groups_fp16 = {int(cases_fp16[0].n_points): cases_fp16}
        else:
            groups_fp16 = _group_by_n_points(cases_fp16)

    stats_by_n_points: dict[str, Any] = {
        "input": input_path.as_posix(),
        "total_cases": int(len(cases)),
        "groups": {},
    }
    for n_points, group in groups.items():
        stats_by_n_points["groups"][str(n_points)] = _compute_group_stats(group, max_percentile=args.percentile)

    stats_fp16_by_n_points: dict[str, Any] | None = None
    if groups_fp16:
        stats_fp16_by_n_points = {
            "input": input_fp16_path.as_posix() if input_fp16_path else "fp16",
            "total_cases": int(len(cases_fp16)) if cases_fp16 else 0,
            "groups": {},
        }
        for n_points, group in groups_fp16.items():
            stats_fp16_by_n_points["groups"][str(n_points)] = _compute_group_stats(group, max_percentile=args.percentile)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats_by_n_points.json").write_text(json.dumps(stats_by_n_points, indent=2), encoding="utf-8")
    if stats_fp16_by_n_points:
        (out_dir / "stats_by_n_points_fp16.json").write_text(
            json.dumps(stats_fp16_by_n_points, indent=2), encoding="utf-8"
        )

    def _print_stats(stats_dict: dict[str, Any], label_prefix: str = ""):
        for n_points, s in stats_dict["groups"].items():
            pcm = s.get("per_case_median_mse", {})
            rk = s.get("eval_relerr_at_mode_k", {})
            sl = s.get("subloss", {})
            et = s.get("eval_time", {})

            def _fmt_stats(name: str, d: dict[str, Any]) -> str:
                med = d.get("p50", float("nan"))
                avg = d.get("mean", float("nan"))
                std = d.get("std", float("nan"))
                p25 = d.get("p25", float("nan"))
                p75 = d.get("p75", float("nan"))
                iqr = p75 - p25
                ci_lo = d.get("mean_ci_lower", float("nan"))
                ci_hi = d.get("mean_ci_upper", float("nan"))
                return f"{name:12s}: median={med:.2e}, mean={avg:.2e}, std={std:.2e}, IQR={iqr:.2e}, CI95=[{ci_lo:.2e}, {ci_hi:.2e}]"

            lbl = f"{label_prefix} " if label_prefix else ""
            print(f"\n[{lbl}N={n_points}] cases={s.get('case_count')} k={s.get('k')}")
            print(f"  {_fmt_stats('Median MSE', pcm)}")
            print(f"  {_fmt_stats('RelErr@k', rk)}")
            print(f"  {_fmt_stats('SubLoss', sl)}")
            print(f"  {_fmt_stats('EvalTime', et)}")

    _print_stats(stats_by_n_points)
    if stats_fp16_by_n_points:
        _print_stats(stats_fp16_by_n_points, label_prefix="FP16")

    if args.no_plot:
        return

    fig_path = out_dir / "median_mse_and_eval_relerr.pdf"
    title = args.title.strip() or None
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
    _plot_summary(
        groups,
        groups_fp16,
        fig_path,
        cfg=cfg,
        title=title,
        show_iqr=bool(args.show_iqr),
        logy_mse=bool(args.logy_mse),
        logy_subloss=bool(args.logy_subloss),
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
