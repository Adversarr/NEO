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
    parser = argparse.ArgumentParser(description="Plot few-shot classification bar chart.")
    
    parser.add_argument("--out_path", type=str, default='/home/adversarr/Repo/g2pt/renders/cls_fewshot.png', help="Output plot path")

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
    parser.add_argument("--height", type=float, default=3.5, help="Figure height")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saving")
    
    # Bar plot specific
    parser.add_argument("--bar_width", type=float, default=0.25, help="Width of bars")
    # Modern Siggraph-style colors (Muted Blue, Muted Red, Muted Green)
    # Alternative: Colorblind friendly palette or Seaborn deep
    # Using a slightly more muted/professional palette
    # Blue (Ours): #4E79A7 (Tableau Blue) or #5DADE2
    # Red: #E15759 (Tableau Red)
    # Green: #76B7B2 (Tableau Teal) or #59A14F (Tableau Green)
    # Let's try: Ours (Blue), Others (Gray/Red/Green)
    # User asked for "modern, siggraph style". Often this means muted but distinct.
    # Let's use a high-contrast but pleasant palette.
    # Method 1 (Ours): #2C3E50 (Dark Blue/Grey) or #E67F0D (Orange highlight)? 
    # Usually 'Ours' is highlighted.
    # Let's stick to the previous hues but better shades:
    # Ours: #3498DB (Flat UI Blue) -> #2980B9
    # Method 2: #E74C3C (Flat UI Red) -> #C0392B
    # Method 3: #2ECC71 (Flat UI Green) -> #27AE60
    # Or maybe more pastel/paper style.
    # Let's use:
    # Ours: #4A90E2 (Nice Blue)
    # M2: #E57373 (Soft Red)
    # M3: #81C784 (Soft Green)
    # Actually, let's go with a very clean "Academic" palette:
    # Ours: #1f77b4 (Matplotlib default blue is actually quite standard) -> Let's use a slightly darker/richer blue #005A9E
    # M2: #D9534F (Bootstrap Danger Red)
    # M3: #5CB85C (Bootstrap Success Green) - maybe too bright.
    # Let's try this set:
    parser.add_argument("--colors", type=str, default="#4E79A7,#F28E2B,#E15759", help="Comma separated hex colors for bars") # Tableau 10: Blue, Orange, Red. 

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
    # Method / Shot | 1 | 3 | 10
    # NEO + PointNet (Ours) | 58.3 | 84.2 | 100.0
    # NeRF-PE + PointNet | 47.5 | 82.5 | 96.7
    # NeRF-PE + PointTransformer | 41.7 | 69.2 | 95.8
    
    shots = ['1-shot', '3-shot', '10-shot']
    methods = [
        'NEO + PointNet (Ours)',
        'NeRF-PE + PointNet',
        'NeRF-PE + PointTransformer'
    ]
    data = np.array([
        [58.3, 84.2, 100.0], # Method 1
        [41.7, 69.2, 95.8],   # Method 3
        [47.5, 82.5, 96.7],  # Method 2
    ])

    # Plotting
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    x = np.arange(len(shots))
    width = args.bar_width
    
    colors = args.colors.split(',')
    if len(colors) < len(methods):
        colors = ["#377EB8", "#E41A1C", "#4DAF4A"] # Fallback

    rects = []
    # Plot bars for each method
    for i in range(len(methods)):
        # Calculate offset for grouped bars
        # Total width of group is len(methods) * width
        # Center is at x
        # Offset for i-th bar: (i - (len(methods)-1)/2) * width
        offset = (i - (len(methods)-1)/2) * width
        rect = ax.bar(x + offset, data[i], width, label=methods[i], color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
        rects.append(rect)
        
        # Add labels on top of bars
        ax.bar_label(rect, padding=3, fmt='%.1f', fontsize=cfg.tick_labelsize * 1.1)

    ax.set_ylabel('Validation Accuracy (%)')
    # ax.set_title('Few-shot Classification on SHREC\'16')
    ax.set_xticks(x)
    ax.set_xticklabels(shots)
    ax.legend(loc='upper left', ncol=1, frameon=False)
    
    # Customize grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Set y limit to make room for labels and legend
    ax.set_ylim(20, 109)

    # Remove top/right spines (already done in setup_style but good to ensure)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=args.dpi)
    print(f"Plot saved to {args.out_path}")

if __name__ == "__main__":
    plot_results()
