import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from matplotlib import font_manager

# --- Style Configuration (Copied from plot_heat_time_poisson.py/scripts/full.py) ---

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

def parse_args():
    parser = argparse.ArgumentParser(description="Plot segmentation mIoU bar chart.")
    
    parser.add_argument("--out_path", type=str, default='/home/adversarr/Repo/g2pt/renders/seg_human.png', help="Output plot path")

    # Style options
    parser.add_argument("--font", type=str, default="Linux Libertine")
    parser.add_argument("--font_size", type=float, default=10.0)
    parser.add_argument("--axes_labelsize", type=float, default=10.0)
    parser.add_argument("--tick_labelsize", type=float, default=10.0)
    parser.add_argument("--legend_fontsize", type=float, default=9.0)
    parser.add_argument("--title_size", type=float, default=12.0)
    parser.add_argument("--line_width", type=float, default=1.5)
    parser.add_argument("--marker_size", type=float, default=5.0)
    
    parser.add_argument("--width", type=float, default=6.0, help="Figure width")
    parser.add_argument("--height", type=float, default=4.0, help="Figure height")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saving")
    
    # Bar plot specific
    parser.add_argument("--bar_width", type=float, default=0.25, help="Width of bars")
    # Modern Siggraph-style colors (Muted Blue, Muted Orange, Muted Red)
    parser.add_argument("--colors", type=str, default="#4E79A7,#F28E2B,#E15759", help="Comma separated hex colors for bars") 

    return parser.parse_args()

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
    
    setup_style(cfg)

    # Data from the table
    # Method / Epochs | 100 | 300
    # NEO (F) + PointNet | 0.80 | 0.82
    # NeRF-PE + PointNet | 0.77 | 0.80
    # NeRF-PE + PointTransformer | 0.78 | 0.80
    
    epochs = ['100 Epochs', '300 Epochs']
    methods = [
        'NEO + PointNet (Ours)',
        'NeRF-PE + PointNet',
        'NeRF-PE + PointTransformer'
    ]
    # Convert to percentage for better visualization? 
    # Or keep as 0-1. Let's keep as 0-1 but format axis.
    # Actually percentage is often easier to read for these small diffs.
    # Let's use percentage.
    data = np.array([
        [0.80, 0.82], # Method 1
        [0.77, 0.80], # Method 2
        [0.78, 0.80]  # Method 3
    ]) * 100.0

    # Plotting
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    x = np.arange(len(epochs))
    width = args.bar_width
    
    colors = args.colors.split(',')
    if len(colors) < len(methods):
        colors = ["#4E79A7", "#F28E2B", "#E15759"] # Fallback

    rects = []
    # Plot bars for each method
    for i in range(len(methods)):
        # Calculate offset for grouped bars
        offset = (i - (len(methods)-1)/2) * width
        rect = ax.bar(x + offset, data[i], width, label=methods[i], color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
        rects.append(rect)
        
        # Add labels on top of bars
        ax.bar_label(rect, padding=3, fmt='%.1f', fontsize=cfg.tick_labelsize * 1.1)

    ax.set_ylabel('mIoU (%)')
    # ax.set_title('Segmentation on Human Body')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(loc='upper left', ncol=1, frameon=False)
    
    # Customize grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Set y limit to zoom in on the relevant range
    # Range is 77 - 82.
    # 70 to 85 seems reasonable.
    ax.set_ylim(74, 85)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=args.dpi)
    print(f"Plot saved to {args.out_path}")

if __name__ == "__main__":
    plot_results()
