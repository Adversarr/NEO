"""
python -u scripts/eig/demo_plot_ambiguity.py --out_dir tmp/ambiguity_mesh_pretty_test --sphere_subdiv 2 --sphere_eig_k 8 --mesh_target_faces 2000 --mesh_eig_k 15 --mesh_eig_index 6 --render_resolution 1000 1000 --render_samples 128 --render_cam_pos 0 1 2 --render_world_up 0 1 0 --render_area_light_pos -1 4 8 --render_area_light_energy=250 --title_fontsize 16 --ylabel_fontsize 15 --sphere_scale 0.765 --font=libertine --cat_cam_pos -0.5 0.6 1 --embed_format jpg --embed_quality=50 --fig_size 10 4.83 --info_fontsize=10.4"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import scipy.sparse.linalg as spla
import trimesh

from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian


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
            # "figure.constrained_layout.use": True, # Conflict with tight_layout
        }
    )


def _colormap_u8(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = float(vmax - vmin)
    if denom <= 0:
        t = np.zeros_like(vals)
    else:
        t = (vals - float(vmin)) / denom
    t = np.clip(t, 0.0, 1.0)
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    rgba = cmap_obj(t)
    return (rgba[:, :3] * 255.0).round().astype(np.uint8)


def _percentile_abs_vmax(arr: np.ndarray, q: float = 99.0) -> float:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = np.abs(x[np.isfinite(x)])
    if x.size == 0:
        return 1.0
    vmax = float(np.percentile(x, q))
    return max(vmax, 1e-8)


def _write_colored_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors_u8: np.ndarray,
) -> None:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    c = np.asarray(colors_u8, dtype=np.uint8)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"colors must be (N,3), got {c.shape}")
    if v.shape[0] != c.shape[0]:
        raise ValueError(f"N mismatch: vertices {v.shape[0]} vs colors {c.shape[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=c)
    mesh.export(path.as_posix())


def _render_mesh_with_blender(
    blender_bin: str,
    blender_script: Path,
    ply_path: Path,
    png_path: Path,
    extra: list[str],
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_bin,
        "-b",
        "-P",
        blender_script.as_posix(),
        "--",
        "--input",
        ply_path.as_posix(),
        "--output",
        png_path.as_posix(),
    ]
    cmd.extend(extra)
    subprocess.run(cmd, check=True)


def _eigsh_mesh(vertices: np.ndarray, faces: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    L, M = mesh_laplacian(vertices, faces)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(int(L.shape[0])).astype(np.float64)
    try:
        evals, evecs = spla.eigsh(L, M=M, k=k, sigma=-1e-6, which="LM", v0=v0)
    except Exception:
        evals, evecs = spla.eigsh(L, M=M, k=k, which="SM", v0=v0)
    evals = np.asarray(evals, dtype=np.float64).reshape(-1)
    evecs = np.asarray(evecs, dtype=np.float64)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def _first_degenerate_block(evals: np.ndarray, start: int = 1, min_size: int = 2, tol: float = 1e-6) -> tuple[int, int]:
    ev = np.asarray(evals, dtype=np.float64).reshape(-1)
    n = int(ev.size)
    if n <= start + min_size:
        raise ValueError("Not enough eigenvalues to find a degenerate block.")

    i = int(start)
    while i < n:
        j = i + 1
        while j < n and abs(ev[j] - ev[i]) <= max(tol, 1e-6 * max(1.0, abs(ev[i]))):
            j += 1
        if (j - i) >= min_size:
            return i, j
        i = j
    raise ValueError("No degenerate eigenvalue block found.")


def _find_closest_pair(evals: np.ndarray, start: int = 1, k_search: int = 20) -> tuple[int, int]:
    ev = np.asarray(evals, dtype=np.float64).reshape(-1)
    n = int(ev.size)
    end = min(n, start + k_search)
    best_pair = (-1, -1)
    min_diff = float("inf")

    for i in range(start, end):
        for j in range(i + 1, end):
            diff = abs(ev[i] - ev[j])
            if diff < min_diff:
                min_diff = diff
                best_pair = (i, j)
    
    if best_pair[0] == -1:
        return (start, start + 1)
    return best_pair


def _format_two_line_title(title: str) -> str:
    s = str(title).strip()
    if not s:
        return s
    i = s.find("(")
    j = s.rfind(")")
    if i == -1 or j == -1 or j <= i:
        return s
    geom = s[:i].strip()
    situation = s[i : j + 1].strip()
    if not geom or not situation:
        return s
    return f"{geom}\n{situation}"


def _crop_image_border(im: np.ndarray, tol: float = 0.03, pad_px: int = 6) -> np.ndarray:
    arr = np.asarray(im)
    if arr.ndim < 2:
        return arr
    if arr.ndim == 2:
        return arr

    arr_f = arr.astype(np.float32)
    if float(np.nanmax(arr_f)) > 1.5:
        arr_f = arr_f / 255.0

    h, w = int(arr_f.shape[0]), int(arr_f.shape[1])
    if h <= 2 or w <= 2:
        return arr

    if arr_f.shape[2] >= 4:
        alpha = arr_f[..., 3]
        if float(np.nanmin(alpha)) < 0.99:
            mask = alpha > 0.01
        else:
            bg = arr_f[0, 0, :3]
            mask = np.any(np.abs(arr_f[..., :3] - bg[None, None, :]) > float(tol), axis=2)
    else:
        bg = arr_f[0, 0, :3]
        mask = np.any(np.abs(arr_f[..., :3] - bg[None, None, :]) > float(tol), axis=2)

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return arr

    y0 = max(int(ys.min()) - int(pad_px), 0)
    y1 = min(int(ys.max()) + int(pad_px) + 1, h)
    x0 = max(int(xs.min()) - int(pad_px), 0)
    x1 = min(int(xs.max()) + int(pad_px) + 1, w)
    return arr[y0:y1, x0:x1, ...]


def _pad_image_to_scale(im: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return im
    if scale <= 0.0:
        return im

    # If scale < 1.0, we want the object (im) to appear smaller.
    # This means the final image size should be larger than im,
    # such that im occupies `scale` fraction of the new size.
    # new_size * scale = old_size  =>  new_size = old_size / scale
    # If scale > 1.0, we want object larger (crop it?), but usually
    # we just want to shrink the object (add padding).
    # We will support scale < 1.0 (add padding) and scale > 1.0 (crop center).

    h, w = im.shape[:2]
    new_h = int(round(h / scale))
    new_w = int(round(w / scale))

    if scale < 1.0:
        # Add padding
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Create new image with white/transparent background
        if im.shape[2] == 4:
            # RGBA: transparent background
            new_im = np.zeros((new_h, new_w, 4), dtype=im.dtype)
        else:
            # RGB: white background
            new_im = np.ones((new_h, new_w, 3), dtype=im.dtype)
            if np.issubdtype(im.dtype, np.floating):
                # if float, 1.0 is white
                pass
            elif np.issubdtype(im.dtype, np.integer):
                new_im = new_im * 255

        new_im[pad_top : pad_top + h, pad_left : pad_left + w] = im
        return new_im
    else:
        # scale > 1.0: Crop center
        # We want to keep only the center part that corresponds to `scale` size?
        # No, "scale > 1.0" visually means object is bigger.
        # This implies we crop into the object.
        # This might be risky if we cut off the object.
        # Let's support it anyway.
        if new_h >= h or new_w >= w:
            return im
        
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        return im[start_h : start_h + new_h, start_w : start_w + new_w]


def _pad_image_top_right(im: np.ndarray, pad_top_px: int, pad_right_px: int) -> np.ndarray:
    pad_top_px = int(max(0, pad_top_px))
    pad_right_px = int(max(0, pad_right_px))
    if pad_top_px == 0 and pad_right_px == 0:
        return im

    h, w = im.shape[:2]
    new_h = h + pad_top_px
    new_w = w + pad_right_px

    if im.ndim == 2:
        new_im = np.ones((new_h, new_w), dtype=im.dtype)
        if np.issubdtype(im.dtype, np.integer):
            new_im = new_im * 255
        new_im[pad_top_px : pad_top_px + h, 0:w] = im
        return new_im

    c = int(im.shape[2])
    if c == 4:
        new_im = np.zeros((new_h, new_w, 4), dtype=im.dtype)
    else:
        new_im = np.ones((new_h, new_w, c), dtype=im.dtype)
        if np.issubdtype(im.dtype, np.integer):
            new_im = new_im * 255
    new_im[pad_top_px : pad_top_px + h, 0:w, ...] = im
    return new_im


def _plot_png(
    ax: plt.Axes,
    png_path: Path,
    title: str,
    title_fontsize: float = 10,
    title_pad: float = 6,
    scale: float = 1.0,
    info_text: str = "",
    info_fontsize: float = 0.0,
    info_pos: tuple[float, float] = (0.98, 0.98),
    info_pad_px: tuple[int, int] = (0, 0),
    info_box_alpha: float = 0.8,
) -> None:
    im = plt.imread(png_path.as_posix())
    im = _crop_image_border(im)
    if scale != 1.0:
        im = _pad_image_to_scale(im, scale)
    if info_text:
        im = _pad_image_top_right(im, pad_top_px=int(info_pad_px[0]), pad_right_px=int(info_pad_px[1]))
    ax.imshow(im)
    ax.set_title(_format_two_line_title(title), fontsize=title_fontsize, pad=title_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if info_text:
        fs = float(info_fontsize) if float(info_fontsize) > 0 else float(title_fontsize) * 0.85
        x, y = float(info_pos[0]), float(info_pos[1])
        ax.text(
            x,
            y,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fs,
            bbox=dict(boxstyle="round", facecolor="white", alpha=float(info_box_alpha), edgecolor="none", pad=0.3),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tmp/ambiguity_demo_mesh", help="Output directory.")
    parser.add_argument("--output", type=str, default="", help="Output 2x4 image path (png/pdf).")
    parser.add_argument("--embed_format", type=str, default="", help="If set, compress renders before embedding into PDF (jpg/webp).")
    parser.add_argument("--embed_quality", type=int, default=85, help="Compression quality (1-100) for embedded renders.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sphere_subdiv", type=int, default=5)
    parser.add_argument("--sphere_eig_k", type=int, default=12)
    parser.add_argument("--sphere_combo_seed", type=int, default=1)
    parser.add_argument(
        "--sphere_scale",
        type=float,
        default=1.0,
        help="Visual scale for sphere (scales the image padding after cropping).",
    )

    parser.add_argument("--mesh", type=str, default="ldata/featuring_models/stanford-bunny.obj")
    parser.add_argument("--mesh_target_faces", type=int, default=35000)
    parser.add_argument("--mesh_eig_k", type=int, default=30)
    parser.add_argument("--mesh_eig_index", type=int, default=6)
    parser.add_argument("--cat_mesh", type=str, default="ldata/pyFM-off/cat-00.off")
    parser.add_argument("--cat_cam_pos", nargs=3, type=float, default=None, help="Camera position for cat mesh.")
    parser.add_argument(
        "--cat_render_resolution",
        nargs=2,
        type=int,
        default=None,
        help="Render resolution for cat mesh only (overrides --render_resolution).",
    )

    parser.add_argument("--fig_size", nargs=2, type=float, default=[12.0, 5.8])
    parser.add_argument("--fig_dpi", type=int, default=300)

    parser.add_argument("--title_fontsize", type=float, default=10.0, help="Font size for subplot titles.")
    parser.add_argument("--title_pad", type=float, default=2.0, help="Padding for subplot titles.")
    parser.add_argument("--ylabel_fontsize", type=float, default=11.0, help="Font size for y-axis labels.")
    parser.add_argument("--ylabel_labelpad", type=float, default=10.0, help="Label padding for y-axis labels.")
    parser.add_argument("--tight_layout_pad", type=float, default=0.2, help="Padding for tight_layout.")
    parser.add_argument("--tight_layout_w_pad", type=float, default=0.15, help="Width padding for tight_layout.")
    parser.add_argument("--tight_layout_h_pad", type=float, default=0.25, help="Height padding for tight_layout.")
    parser.add_argument("--info_fontsize", type=float, default=0.0, help="Font size for Idx/Val (0=auto).")
    parser.add_argument("--info_pos", nargs=2, type=float, default=[0.98, 0.98], help="(x,y) in axes coords for Idx/Val.")
    parser.add_argument(
        "--info_pad_px",
        nargs=2,
        type=int,
        default=[50, 50],
        help="Extra (top,right) padding in pixels inside each subplot for Idx/Val.",
    )
    parser.add_argument("--info_box_alpha", type=float, default=0.8, help="Alpha for Idx/Val background box.")

    parser.add_argument("--blender", type=str, default="blender")
    parser.add_argument(
        "--render_script",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "renders" / "render_mesh_once.py").as_posix()),
    )
    parser.add_argument("--render_resolution", nargs=2, type=int, default=[1000, 1000])
    parser.add_argument("--render_samples", type=int, default=128)
    parser.add_argument("--render_engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--render_world_strength", type=float, default=0.15)
    parser.add_argument("--render_background_rgb", nargs=3, type=float, default=None)
    parser.add_argument("--render_area_light_energy", type=float, default=500.0)
    parser.add_argument("--render_point_light_energy", type=float, default=200.0)
    parser.add_argument("--render_roughness", type=float, default=0.7)
    parser.add_argument("--render_specular", type=float, default=0.25)
    parser.add_argument("--render_cam_pos", nargs=3, type=float, default=[0.0, -5.0, 1.0])
    parser.add_argument("--render_cam_target", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--render_area_light_pos", nargs=3, type=float, default=[1.0, 4.0, -8.0])
    parser.add_argument("--render_subdivision", type=int, default=1)
    parser.add_argument("--render_world_up", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--font", type=str, default="libertine", help="Preferred serif font family (e.g., libertine).")
    parser.add_argument("render_args", nargs=argparse.REMAINDER, help="Extra args for render_mesh_once.py (after --)")
    args = parser.parse_args()

    setup_style(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plys_dir = out_dir / "ply"
    renders_dir = out_dir / "renders"
    plys_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    compressed_renders_dir = renders_dir / "compressed"
    compressed_renders_dir.mkdir(parents=True, exist_ok=True)

    render_extra_args = list(args.render_args)
    if render_extra_args and render_extra_args[0] == "--":
        render_extra_args = render_extra_args[1:]

    render_common = [
        "--engine",
        str(args.render_engine),
        "--resolution",
        str(int(args.render_resolution[0])),
        str(int(args.render_resolution[1])),
        "--samples",
        str(int(args.render_samples)),
        "--subdivision",
        str(int(args.render_subdivision)),
        "--world_strength",
        str(float(args.render_world_strength)),
        "--area_light_energy",
        str(float(args.render_area_light_energy)),
        "--point_light_energy",
        str(float(args.render_point_light_energy)),
        "--roughness",
        str(float(args.render_roughness)),
        "--specular",
        str(float(args.render_specular)),
        "--cam_pos",
        str(float(args.render_cam_pos[0])),
        str(float(args.render_cam_pos[1])),
        str(float(args.render_cam_pos[2])),
        "--world_up",
        str(float(args.render_world_up[0])),
        str(float(args.render_world_up[1])),
        str(float(args.render_world_up[2])),
        "--cam_target",
        str(float(args.render_cam_target[0])),
        str(float(args.render_cam_target[1])),
        str(float(args.render_cam_target[2])),
        "--lights_target_center",
        "--area_light_pos",
        str(float(args.render_area_light_pos[0])),
        str(float(args.render_area_light_pos[1])),
        str(float(args.render_area_light_pos[2])),
    ]
    if args.render_background_rgb is not None:
        render_common.extend(
            [
                "--background_rgb",
                str(float(args.render_background_rgb[0])),
                str(float(args.render_background_rgb[1])),
                str(float(args.render_background_rgb[2])),
            ]
        )
    render_common.extend(render_extra_args)

    script = Path(args.render_script)

    sphere = trimesh.creation.icosphere(subdivisions=int(args.sphere_subdiv), radius=0.7)
    sphere_v = np.asarray(sphere.vertices, dtype=np.float64)
    sphere_f = np.asarray(sphere.faces, dtype=np.int64)
    sphere_evals, sphere_evecs = _eigsh_mesh(sphere_v, sphere_f, k=int(args.sphere_eig_k), seed=int(args.seed))
    d0, d1 = _first_degenerate_block(sphere_evals, start=1, min_size=3)
    U = sphere_evecs[:, d0:d1]
    run_a_1 = U[:, 0]
    rng = np.random.default_rng(int(args.sphere_combo_seed))
    coeff = rng.standard_normal(U.shape[1]).astype(np.float64)
    coeff /= max(float(np.linalg.norm(coeff)), 1e-12)
    run_b_1 = U @ coeff
    vmax_1 = _percentile_abs_vmax(np.concatenate([run_a_1, run_b_1]), q=99.0)

    mesh_path = Path(args.mesh)
    bunny_tm = trimesh.load(mesh_path.as_posix(), process=False)
    if isinstance(bunny_tm, trimesh.Scene):
        bunny_tm = trimesh.util.concatenate(tuple(bunny_tm.geometry.values()))
    if not isinstance(bunny_tm, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh: {mesh_path}")
    target_faces = int(args.mesh_target_faces)
    if target_faces > 0 and bunny_tm.faces.shape[0] > target_faces:
        try:
            bunny_tm = bunny_tm.simplify_quadric_decimation(face_count=target_faces)
        except TypeError:
            bunny_tm = bunny_tm.simplify_quadric_decimation(target_faces)
    bunny_v = np.asarray(bunny_tm.vertices, dtype=np.float64)
    bunny_f = np.asarray(bunny_tm.faces, dtype=np.int64)

    # Normalize bunny to unit radius
    bunny_v -= np.mean(bunny_v, axis=0)
    bunny_v /= np.max(np.linalg.norm(bunny_v, axis=1))

    bunny_evals, bunny_evecs = _eigsh_mesh(bunny_v, bunny_f, k=int(args.mesh_eig_k), seed=int(args.seed))
    idx = int(args.mesh_eig_index)
    if idx < 1 or idx >= bunny_evecs.shape[1]:
        raise ValueError(f"--mesh_eig_index must be in [1, {bunny_evecs.shape[1] - 1}]")
    run_a_2 = bunny_evecs[:, idx]
    run_b_2 = run_a_2.copy()
    run_a_3 = run_a_2
    run_b_3 = -run_a_2

    vmax_23 = _percentile_abs_vmax(np.concatenate([run_a_2, run_b_2, run_a_3, run_b_3]), q=99.5)

    cat_path = Path(args.cat_mesh)
    if not cat_path.exists():
        raise FileNotFoundError(f"Cat mesh not found at: {cat_path}")
    
    cat_tm = trimesh.load(cat_path.as_posix(), process=False)
    if isinstance(cat_tm, trimesh.Scene):
        cat_tm = trimesh.util.concatenate(tuple(cat_tm.geometry.values()))
    
    cat_v = np.asarray(cat_tm.vertices, dtype=np.float64)
    cat_f = np.asarray(cat_tm.faces, dtype=np.int64)
    cat_v -= np.mean(cat_v, axis=0)
    cat_v /= np.max(np.linalg.norm(cat_v, axis=1))

    cat_evals, cat_evecs = _eigsh_mesh(cat_v, cat_f, k=int(args.mesh_eig_k), seed=int(args.seed))
    cat_idx_a, cat_idx_b = _find_closest_pair(cat_evals, start=1, k_search=20)
    run_a_4 = cat_evecs[:, cat_idx_a]
    run_b_4 = cat_evecs[:, cat_idx_b]
    vmax_4 = _percentile_abs_vmax(np.concatenate([run_a_4, run_b_4]), q=99.5)

    info_1_a = (d0, sphere_evals[d0])
    info_1_b = (d0, sphere_evals[d0])
    info_2_a = (idx, bunny_evals[idx])
    info_2_b = (idx, bunny_evals[idx])
    info_3_a = (idx, bunny_evals[idx])
    info_3_b = (idx, bunny_evals[idx])
    info_4_a = (cat_idx_a, cat_evals[cat_idx_a])
    info_4_b = (cat_idx_b, cat_evals[cat_idx_b])

    cat_cam = args.cat_cam_pos
    cat_res = args.cat_render_resolution

    cases = [
        ("sphere_rotate", sphere_v, sphere_f, run_a_1, run_b_1, vmax_1, info_1_a, info_1_b, None, None),
        ("bunny_match", bunny_v, bunny_f, run_a_2, run_b_2, vmax_23, info_2_a, info_2_b, None, None),
        ("bunny_signflip", bunny_v, bunny_f, run_a_3, run_b_3, vmax_23, info_3_a, info_3_b, None, None),
        ("cat_near_degenerate", cat_v, cat_f, run_a_4, run_b_4, vmax_4, info_4_a, info_4_b, cat_cam, cat_res),
    ]

    embed_fmt = str(args.embed_format).lstrip(".").lower() if args.embed_format else ""
    embed_quality = int(args.embed_quality)
    compress_script = (Path(__file__).resolve().parents[1] / "compress_png.py").resolve() if embed_fmt else None

    png_grid: list[list[tuple[Path, tuple]]] = [[], []]
    for col, (tag, v, f, va, vb, vmax, info_a, info_b, cam_pos, res) in enumerate(cases):
        case_render_args = list(render_common)
        if cam_pos is not None:
            case_render_args.extend(
                [
                    "--cam_pos",
                    str(float(cam_pos[0])),
                    str(float(cam_pos[1])),
                    str(float(cam_pos[2])),
                ]
            )
        if res is not None:
            case_render_args.extend(
                [
                    "--resolution",
                    str(int(res[0])),
                    str(int(res[1])),
                ]
            )

        for row, (run_tag, vals, info) in enumerate([("run_a", va, info_a), ("run_b", vb, info_b)]):
            ply_path = plys_dir / f"{tag}_{run_tag}.ply"
            png_path = renders_dir / f"{tag}_{run_tag}.png"
            colors = _colormap_u8(vals, "RdBu_r", vmin=-float(vmax), vmax=float(vmax))
            _write_colored_mesh_ply(ply_path, v, f, colors)
            _render_mesh_with_blender(args.blender, script, ply_path, png_path, case_render_args)
            plot_path = png_path
            if embed_fmt and compress_script is not None:
                compressed_path = compressed_renders_dir / f"{tag}_{run_tag}.{embed_fmt}"
                cmd = [
                    "python",
                    compress_script.as_posix(),
                    "--input",
                    png_path.as_posix(),
                    "--output",
                    compressed_path.as_posix(),
                    "--format",
                    embed_fmt,
                    "--quality",
                    str(int(embed_quality)),
                ]
                subprocess.run(cmd, check=True)
                plot_path = compressed_path
            png_grid[row].append((plot_path, info))

    fig, axes = plt.subplots(2, 4, figsize=(args.fig_size[0], args.fig_size[1]), dpi=args.fig_dpi)
    fig.patch.set_facecolor("white")
    col_titles = [
        "Sphere (Degenerate Rotate)",
        "Bunny (Lucky Match)",
        "Bunny (Sign Flip)",
        "Cat (Nearly Degenerate)",
    ]
    for c in range(4):
        axes[0, c].set_facecolor("white")
        axes[1, c].set_facecolor("white")
        scale = float(args.sphere_scale) if c == 0 else 1.0
        
        path_a, (idx_a, val_a) = png_grid[0][c]
        path_b, (idx_b, val_b) = png_grid[1][c]

        txt_a = f"Eigen Index: {idx_a}\nVal: {val_a:.2e}"
        txt_b = f"Eigen Index: {idx_b}\nVal: {val_b:.2e}"

        _plot_png(
            axes[0, c],
            path_a,
            col_titles[c],
            title_fontsize=args.title_fontsize,
            title_pad=args.title_pad,
            scale=scale,
            info_text=txt_a,
            info_fontsize=args.info_fontsize,
            info_pos=(float(args.info_pos[0]), float(args.info_pos[1])),
            info_pad_px=(int(args.info_pad_px[0]), int(args.info_pad_px[1])),
            info_box_alpha=args.info_box_alpha,
        )
        _plot_png(
            axes[1, c],
            path_b,
            "",
            title_fontsize=args.title_fontsize,
            title_pad=args.title_pad,
            scale=scale,
            info_text=txt_b,
            info_fontsize=args.info_fontsize,
            info_pos=(float(args.info_pos[0]), float(args.info_pos[1])),
            info_pad_px=(int(args.info_pad_px[0]), int(args.info_pad_px[1])),
            info_box_alpha=args.info_box_alpha,
        )

    axes[0, 0].set_ylabel("Run A", fontsize=args.ylabel_fontsize, labelpad=args.ylabel_labelpad)
    axes[1, 0].set_ylabel("Run B", fontsize=args.ylabel_fontsize, labelpad=args.ylabel_labelpad)

    fig.tight_layout(pad=args.tight_layout_pad, w_pad=args.tight_layout_w_pad, h_pad=args.tight_layout_h_pad)
    out_path = Path(args.output) if args.output else (out_dir / "ambiguity_compare_2x4.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
