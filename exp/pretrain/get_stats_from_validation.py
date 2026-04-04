from __future__ import annotations

import argparse
import json
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


def setup_style(font: str) -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

    serif_fonts = _pick_serif_fonts(font)
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
            "lines.markersize": 4,
            "axes.grid": False,
            "figure.constrained_layout.use": True,
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


def _load_cases(path: Path, k: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list JSON at {path}")

    n_points_list = []
    scores_list = []
    relerr_list = []
    subloss_list = []

    for d in raw:
        if not isinstance(d, dict):
            continue
        n_points = d.get("n_points")
        if not isinstance(n_points, int) or n_points <= 0:
            continue
        scores = d.get("scores", {}).get("pointcloud_vs_network")
        if scores is None:
            continue
        scores_arr = _as_float_array(scores).reshape(-1)
        k_picked = int(scores_arr.size) if k is None else int(k)
        scores_arr = _align_length(scores_arr, k_picked, fill_value=float("nan"))

        relerr = d.get("eval_relerr", [])
        relerr_arr = _as_float_array(relerr).reshape(-1)
        if relerr_arr.size == k_picked - 1:
            relerr_arr = np.concatenate([np.asarray([float("nan")], dtype=np.float64), relerr_arr], axis=0)
        relerr_arr = _align_length(relerr_arr, k_picked, fill_value=float("nan"))

        try:
            subloss_f = float(d.get("subloss"))
        except Exception:
            subloss_f = float("nan")

        n_points_list.append(int(n_points))
        scores_list.append(scores_arr)
        relerr_list.append(relerr_arr)
        subloss_list.append(subloss_f)

    if not scores_list:
        raise ValueError(f"No valid cases found in {path}")

    n_points_arr = np.asarray(n_points_list, dtype=np.int64)
    scores_mat = np.stack(scores_list, axis=0)
    relerr_mat = np.stack(relerr_list, axis=0)
    subloss_arr = np.asarray(subloss_list, dtype=np.float64)
    return n_points_arr, scores_mat, relerr_mat, subloss_arr


def _pick_group(n_points: np.ndarray, scores: np.ndarray, relerr: np.ndarray, subloss: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = n_points == int(n)
    if not np.any(mask):
        raise ValueError(f"No cases with n_points={n}")
    return scores[mask], relerr[mask], subloss[mask]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="outputs.json")
    p.add_argument("--out", type=str, default="validation_siggraph.pdf")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--n_points", type=int, default=0)
    p.add_argument("--font", type=str, default="Linux Libertine")
    p.add_argument("--logy_relerr", action="store_true")
    p.add_argument("--logy_mse", action="store_true")
    p.add_argument("--show_iqr", action="store_true")
    p.add_argument("--title", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    k = int(args.k) if int(args.k) > 0 else None
    n_points, scores, relerr, subloss = _load_cases(input_path, k=k)

    n_pick = int(args.n_points) if int(args.n_points) > 0 else int(np.max(n_points))
    scores_g, relerr_g, subloss_g = _pick_group(n_points, scores, relerr, subloss, n_pick)

    scores_g = np.clip(scores_g, -1.0, 1.0)
    mse_g = 1.0 - scores_g

    mse_med = np.median(mse_g, axis=0)
    mse_p25 = np.percentile(mse_g, 25, axis=0)
    mse_p75 = np.percentile(mse_g, 75, axis=0)
    rel_med = np.nanmedian(relerr_g[:, 1:], axis=0) if relerr_g.shape[1] >= 2 else np.asarray([], dtype=np.float64)

    setup_style(args.font)
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6.8, 4.8), sharex=True, constrained_layout=True)
    x = np.arange(1, mse_med.size + 1)

    if args.logy_mse:
        mse_med = np.maximum(mse_med, 1e-15)
        mse_p25 = np.maximum(mse_p25, 1e-15)
        mse_p75 = np.maximum(mse_p75, 1e-15)

    ax0.plot(x, mse_med, color="C0", label=f"N={n_pick} (n={scores_g.shape[0]})", marker="o", markeredgewidth=0.0)
    if args.show_iqr:
        ax0.fill_between(x, mse_p25, mse_p75, color="C0", alpha=0.18, linewidth=0.0)
    ax0.set_ylabel("Median MSE (1 - score)")
    ax0.legend(loc="upper right", frameon=True)
    if args.logy_mse:
        ax0.set_yscale("log")
        ax0.set_ylim(bottom=1e-6)
    else:
        ax0.set_ylim(bottom=0.0)

    if rel_med.size:
        x_rel = np.arange(2, mse_med.size + 1)
        ax1.plot(x_rel, rel_med, color="C1", marker="o", markeredgewidth=0.0)
    ax1.set_xlabel("Mode i")
    ax1.set_ylabel("Median eval rel. error")
    ax1.set_ylim(bottom=1e-4)
    if args.logy_relerr:
        ax1.set_yscale("log")

    title = args.title.strip()
    if title:
        fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    per_case_median_mse = np.median(mse_g, axis=1)
    relerr_ex0 = relerr_g[:, 1:] if relerr_g.shape[1] >= 2 else relerr_g[:, :0]
    relerr_ex0_flat = relerr_ex0.reshape(-1)
    relerr_ex0_finite = relerr_ex0_flat[np.isfinite(relerr_ex0_flat)]
    subloss_finite = subloss_g[np.isfinite(subloss_g)]
    print(
        json.dumps(
            {
                "input": input_path.as_posix(),
                "n_points": int(n_pick),
                "cases": int(scores_g.shape[0]),
                "k": int(mse_med.size),
                "per_case_median_mse": {
                    "p50": float(np.percentile(per_case_median_mse, 50)),
                    "p90": float(np.percentile(per_case_median_mse, 90)),
                    "p95": float(np.percentile(per_case_median_mse, 95)),
                },
                "mse_at_mode_k": {
                    "p95": float(np.percentile(mse_g[:, -1], 95)),
                    "max": float(np.max(mse_g[:, -1])),
                },
                "eval_relerr_at_mode_k": {
                    "p95": float(np.percentile(relerr_g[:, -1][np.isfinite(relerr_g[:, -1])], 95))
                    if np.any(np.isfinite(relerr_g[:, -1]))
                    else float("nan"),
                    "max": float(np.max(relerr_g[:, -1][np.isfinite(relerr_g[:, -1])]))
                    if np.any(np.isfinite(relerr_g[:, -1]))
                    else float("nan"),
                },
                "eval_relerr_all_modes_excl0": {
                    "p50": float(np.percentile(relerr_ex0_finite, 50)) if relerr_ex0_finite.size else float("nan"),
                    "p95": float(np.percentile(relerr_ex0_finite, 95)) if relerr_ex0_finite.size else float("nan"),
                    "mean": float(np.mean(relerr_ex0_finite)) if relerr_ex0_finite.size else float("nan"),
                },
                "subloss": {
                    "p50": float(np.percentile(subloss_finite, 50)) if subloss_finite.size else float("nan"),
                    "p95": float(np.percentile(subloss_finite, 95)) if subloss_finite.size else float("nan"),
                    "mean": float(np.mean(subloss_finite)) if subloss_finite.size else float("nan"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
