import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from matplotlib import font_manager

# --- Style Configuration (Adapted from scripts/full.py) ---

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
            "text.latex.preamble": r"\usepackage{amsfonts}",
            "mathtext.fontset": "cm",
        }
    )

# --- Data Loading and Plotting ---

def load_data(filepath, time_offset=0.0):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    points = []
    for entry in data:
        try:
            poisson_stats = entry['pcg']['poisson']
            time = poisson_stats['time_total'] + time_offset
            num_iter = poisson_stats['num_iter']
            rel_res = poisson_stats['rel_res']
            tol = entry['params']['tol']
            points.append({'time': time, 'iters': num_iter, 'rel_res': rel_res, 'tol': tol})
        except KeyError:
            continue
    
    points.sort(key=lambda x: x['rel_res'], reverse=True)
    return points

def parse_args():
    parser = argparse.ArgumentParser(description="Plot heat time vs poisson residual.")
    
    # Files
    parser.add_argument("--ref_file", type=str, default='/home/adversarr/Repo/g2pt/heat-rocker-arm-ichol.json', help="Path to reference JSON file")
    parser.add_argument("--def_file", type=str, default='/home/adversarr/Repo/g2pt/heat-rocker-arm-additive-16.json', help="Path to deflated PCG JSON file")
    parser.add_argument("--out_path", type=str, default='/home/adversarr/Repo/g2pt/renders/heat_time_poisson.png', help="Output plot path")
    parser.add_argument("--out_path_iter", type=str, default="", help="Output plot path for iteration-count x-axis (default: derive from --out_path)")

    # Time offset option
    parser.add_argument("--deflation_time_offset", type=float, default=0.0, help="Additional time to add to Deflated PCG results (default: 0.0)")

    # Style options (from full.py)
    parser.add_argument("--font", type=str, default="Linux Libertine")
    parser.add_argument("--font_size", type=float, default=10.0)
    parser.add_argument("--axes_labelsize", type=float, default=10.0)
    parser.add_argument("--tick_labelsize", type=float, default=9.0)
    parser.add_argument("--legend_fontsize", type=float, default=9.0)
    parser.add_argument("--title_size", type=float, default=10.0)
    parser.add_argument("--line_width", type=float, default=1.5)
    parser.add_argument("--marker_size", type=float, default=5.0)
    
    parser.add_argument("--width", type=float, default=8.0, help="Figure width")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saving")

    return parser.parse_args()

def _derive_iter_out_path(out_path: str) -> str:
    p = Path(out_path)
    return str(p.with_name(f"{p.stem}-iter{p.suffix}"))

def _plot_panel(
    *,
    ref_x,
    ref_y,
    def_x,
    def_y,
    xlabel: str,
    out_path: str,
    width: float,
    height: float,
    dpi: int,
    cfg: StyleConfig,
    xlim_left: Optional[float] = None,
) -> None:
    # Colors from scripts/full.py
    c_ours_32 = "#377EB8"  # Blue (Ours)
    c_arpack = "#E41A1C"   # Red (Reference)

    plt.figure(figsize=(width, height))

    plt.semilogy(ref_x, ref_y, 's-', color=c_arpack, label='ICPCG',
                 markersize=cfg.marker_size, linewidth=cfg.line_width)
    plt.semilogy(def_x, def_y, 'o-', color=c_ours_32, label='Ours',
                 markersize=cfg.marker_size, linewidth=cfg.line_width)

    plt.xlabel(xlabel)
    plt.ylabel(r'Relative Residual ($\|r\|/\|b\|$)')

    plt.grid(True, which="major", ls="-", alpha=0.6)
    plt.grid(True, which="minor", ls=":", alpha=0.35)
    plt.legend(loc="upper right", frameon=False)
    if xlim_left is not None:
        plt.xlim(left=xlim_left)

    out_dir = Path(out_path).expanduser().resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    print(f"Plot saved to {out_path}")
    plt.close()

def plot_results():
    args = parse_args()

    cfg = StyleConfig(
        font=args.font,
        font_size=args.font_size,
        axes_labelsize=args.axes_labelsize,
        tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize,
        title_size=args.title_size,
        line_width=args.line_width,
        marker_size=args.marker_size
    )
    
    # Setup style
    setup_style(cfg)

    ref_data = load_data(args.ref_file)
    def_data = load_data(args.def_file, time_offset=args.deflation_time_offset)
    
    ref_res = [p['rel_res'] for p in ref_data]
    ref_time = [p['time'] for p in ref_data]
    ref_iters = [p['iters'] for p in ref_data]
    
    def_res = [p['rel_res'] for p in def_data]
    def_time = [p['time'] for p in def_data]
    def_iters = [p['iters'] for p in def_data]
    
    _plot_panel(
        ref_x=ref_time,
        ref_y=ref_res,
        def_x=def_time,
        def_y=def_res,
        xlabel='Time (s)',
        out_path=args.out_path,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        cfg=cfg,
        xlim_left=1e-6,
    )

    out_path_iter = args.out_path_iter.strip() or _derive_iter_out_path(args.out_path)
    _plot_panel(
        ref_x=ref_iters,
        ref_y=ref_res,
        def_x=def_iters,
        def_y=def_res,
        xlabel='Iteration Count',
        out_path=out_path_iter,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        cfg=cfg,
        xlim_left=0.5,
    )

if __name__ == "__main__":
    plot_results()
