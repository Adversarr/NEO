from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import LogFormatterMathtext, LogLocator

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


def setup_style(font: str = "Linux Libertine O", font_size: int = 12) -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

    serif_fonts = _pick_serif_fonts(font)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
            "axes.labelsize": font_size,
            "font.size": font_size,
            "legend.fontsize": font_size - 2,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "grid.color": "#EEEEEE",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def load_data(m_values: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sublosses = []
    evec_mses = []

    root_dir = Path(__file__).resolve().parent.parent

    for m in m_values:
        filename = f"outputs-ours-{m}.json"
        file_path = root_dir / filename

        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping m={m}")
            sublosses.append(np.array([]))
            evec_mses.append(np.array([]))
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            curr_sublosses = [d["subloss"] for d in data if "subloss" in d]
            sublosses.append(np.array(curr_sublosses))

            curr_mses = []
            for d in data:
                scores = (d.get("scores") or {}).get("pointcloud_vs_network")
                if scores:
                    curr_mses.append(1.0 - np.mean(scores))
            evec_mses.append(np.array(curr_mses))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            sublosses.append(np.array([]))
            evec_mses.append(np.array([]))

    return sublosses, evec_mses


def _format_log_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.minorticks_off()


def plot_ablation(
    m_values: list[int],
    out_path: Path,
    font: str,
    font_size: int,
    figsize: tuple[float, float],
    also_png: bool,
) -> None:
    setup_style(font=font, font_size=font_size)

    sublosses_list, evec_mses_list = load_data(m_values)

    # Filter valid m values
    m_plot = []
    sub_medians = []
    sub_q1 = []
    sub_q3 = []
    mse_medians = []
    mse_q1 = []
    mse_q3 = []

    for i, m in enumerate(m_values):
        s_data = sublosses_list[i]
        e_data = evec_mses_list[i]
        if s_data.size > 0 and e_data.size > 0:
            m_plot.append(m)
            sub_medians.append(np.median(s_data))
            sub_q1.append(np.percentile(s_data, 25))
            sub_q3.append(np.percentile(s_data, 75))
            mse_medians.append(np.median(e_data))
            mse_q1.append(np.percentile(e_data, 25))
            mse_q3.append(np.percentile(e_data, 75))

    if not m_plot:
        print("No valid data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    color1 = "#268BD2"
    color2 = "#CB4B16"

    # (a) Subspace Loss
    ax1.set_title("(a) Subspace Loss", loc="left", fontsize=font_size)
    ax1.set_xlabel(r"Subspace dimension $m$")
    ax1.set_ylabel("Subspace Loss")
    ax1.plot(m_plot, sub_medians, marker="o", color=color1, label="Median")
    ax1.fill_between(m_plot, sub_q1, sub_q3, color=color1, alpha=0.2, label="IQR")
    ax1.set_yscale("log")
    _format_log_axis(ax1)
    ax1.grid(True, which="major", axis="y")

    # (b) Eigenvector MSE
    ax2.set_title("(b) Eigenvector MSE", loc="left", fontsize=font_size)
    ax2.set_xlabel(r"Subspace dimension $m$")
    ax2.set_ylabel("Eigenvector MSE")
    ax2.plot(m_plot, mse_medians, marker="s", color=color2, label="Median")
    ax2.fill_between(m_plot, mse_q1, mse_q3, color=color2, alpha=0.2, label="IQR")
    ax2.set_yscale("log")
    _format_log_axis(ax2)
    ax2.grid(True, which="major", axis="y")

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    if also_png:
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[96, 128, 160, 192, 256])
    parser.add_argument("--font", type=str, default="Linux Libertine O")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("ablation.pdf"))
    parser.add_argument("--also-png", action="store_true")
    args = parser.parse_args()

    plot_ablation(
        m_values=list(args.m),
        out_path=args.out,
        font=args.font,
        font_size=args.font_size,
        figsize=(args.width, args.height),
        also_png=bool(args.also_png),
    )

