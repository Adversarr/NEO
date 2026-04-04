import json
import glob
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
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
            "mathtext.fontset": "cm",
        }
    )


def _extract_subloss(content: Mapping[str, Any]) -> float | None:
    scores = content.get("scores", {})
    if isinstance(scores, Mapping):
        loss = scores.get("subspace_loss")
        if loss is not None:
            try:
                return float(loss)
            except Exception:
                return None
    precisions = content.get("precisions")
    if isinstance(precisions, Mapping):
        fp32 = precisions.get("fp32")
        if isinstance(fp32, Mapping):
            loss = fp32.get("loss")
            if loss is None:
                scores = fp32.get("scores", {})
                if isinstance(scores, Mapping):
                    loss = scores.get("subspace_loss")
            if loss is not None:
                try:
                    return float(loss)
                except Exception:
                    return None
    return None


def _collect_results(root_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    pattern = root_path / "tmp_Ours-Small-NoRoPE_*" / "*" / "inferred" / "results.json"
    files = glob.glob(str(pattern))
    print(f"Found {len(files)} result files.")
    for file_path in files:
        try:
            path = Path(file_path)
            n_dir = path.parent.parent
            model_dir = n_dir.parent
            dir_name = model_dir.name
            if not dir_name.startswith("tmp_Ours-Small-NoRoPE_"):
                continue
            model_name = dir_name.replace("tmp_Ours-Small-NoRoPE_", "")
            with open(file_path, "r") as f:
                data = json.load(f)
            n_points = data.get("n_points")
            if n_points is None:
                try:
                    n_points = int(n_dir.name)
                except ValueError:
                    print(f"Could not determine n_points for {file_path}")
                    continue
            subloss = _extract_subloss(data)
            if subloss is None:
                print(f"Warning: No subspace_loss in {file_path}")
                continue
            results[int(n_points)].append(
                {
                    "model": model_name,
                    "subloss": float(subloss),
                    "path": str(path),
                }
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return results


def _print_stats(results: Dict[int, List[Dict[str, Any]]]) -> None:
    sorted_ns = sorted(results.keys())
    print("\nSpanloss Statistics (sorted by Spanloss descending for each N):")
    print("=" * 60)
    for n in sorted_ns:
        cases = results[n]
        cases.sort(key=lambda x: x["subloss"], reverse=True)
        losses = [x["subloss"] for x in cases]
        avg_loss = np.mean(losses)
        median_loss = np.median(losses)
        print(f"\nN = {n} (Count: {len(cases)})")
        print(f"Average Spanloss: {avg_loss:.6f}")
        print(f"Median Spanloss:  {median_loss:.6f}")
        print("-" * 40)
        print(f"{'Model':<30} | {'Spanloss':<15}")
        print("-" * 40)
        for item in cases:
            print(f"{item['model']:<30} | {item['subloss']:.6f}")


def _plot_worst_curve(
    results: Dict[int, List[Dict[str, Any]]],
    worst_n: int,
    worst_count: int,
    out_dir: Path,
    out_prefix: str,
    formats: Sequence[str],
    dpi: int,
    fig_w: float,
    fig_h: float,
    cfg: StyleConfig,
    title: str | None,
) -> None:
    cases = results.get(int(worst_n), [])
    if not cases:
        print(f"Warning: no data for N={worst_n}")
        return
    cases_sorted = sorted(cases, key=lambda x: x["subloss"], reverse=True)
    if worst_count > 0:
        cases_sorted = cases_sorted[: int(worst_count)]
    y = np.asarray([x["subloss"] for x in cases_sorted], dtype=np.float64)
    x = np.arange(1, len(y) + 1, dtype=np.float64)
    setup_style(cfg)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)
    ax.plot(x, y, "-", color="#377EB8", marker="o", markersize=cfg.marker_size)
    ax.set_xlabel("Rank (worst to best)")
    ax.set_ylabel(r"Span Loss ($\mathcal{E}_\mathrm{sub}(i)$)")
    if title:
        ax.set_title(title)
    ax.set_xlim(1, max(1, int(len(y)))+0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        out_path = out_dir / f"{out_prefix}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyze gallery results.")
    parser.add_argument("--root", type=str, default="ldata/time_test_2", help="Root directory of results")
    parser.add_argument("--plot_worst", action="store_true")
    parser.add_argument("--worst_n", type=int, default=32768)
    parser.add_argument("--worst_count", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="scripts/stats/plots")
    parser.add_argument("--out_prefix", type=str, default="gallery_worst")
    parser.add_argument("--formats", type=str, default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=3.2)
    parser.add_argument("--title", type=str, default="")

    parser.add_argument("--font", type=str, default="Linux Libertine")
    parser.add_argument("--font_size", type=float, default=10.0)
    parser.add_argument("--axes_labelsize", type=float, default=10.0)
    parser.add_argument("--tick_labelsize", type=float, default=9.0)
    parser.add_argument("--legend_fontsize", type=float, default=9.0)
    parser.add_argument("--title_size", type=float, default=10.0)
    parser.add_argument("--line_width", type=float, default=1.5)
    parser.add_argument("--marker_size", type=float, default=5.0)
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: {root_path} does not exist.")
        return

    results = _collect_results(root_path)
    if not results:
        return

    _print_stats(results)

    if not bool(args.plot_worst):
        return

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
    out_dir = (Path(__file__).resolve().parent.parent / str(args.out_dir)).resolve()
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    title = str(args.title).strip() or None
    _plot_worst_curve(
        results=results,
        worst_n=int(args.worst_n),
        worst_count=int(args.worst_count),
        out_dir=out_dir,
        out_prefix=str(args.out_prefix),
        formats=formats,
        dpi=int(args.dpi),
        fig_w=float(args.width),
        fig_h=float(args.height),
        cfg=cfg,
        title=title,
    )

if __name__ == "__main__":
    main()
