import argparse
import json
import subprocess
from pathlib import Path


def _render_with_blender(
    blender_bin: str,
    blender_script: Path,
    input_path: Path,
    output_path: Path,
    extra_args: list[str],
    *,
    dry_run: bool,
):
    cmd = [
        blender_bin,
        "-b",
        "-P",
        blender_script.as_posix(),
        "--",
        "--input",
        input_path.as_posix(),
        "--output",
        output_path.as_posix(),
    ]
    cmd.extend(extra_args)
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Render outputs of scripts/biharmonic_geodesic.py")
    parser.add_argument("--out_dir", type=str, default="output_biharmonic", help="Output directory containing PLYs")
    parser.add_argument("--blender", type=str, default="blender", help="Blender executable")
    parser.add_argument(
        "--render_script_mesh",
        type=str,
        default=str((Path(__file__).resolve().parent / "render_mesh_once.py").as_posix()),
        help="Blender script for mesh rendering",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=[1000, 1000], help="Render resolution W H")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        help="Blender render engine",
    )
    parser.add_argument("--denoise", action="store_true", help="Enable denoising in Cycles")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running Blender")
    parser.add_argument("render_extra", nargs=argparse.REMAINDER, help="Extra args passed to render_mesh_once.py")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    render_dir = out_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    script = Path(args.render_script_mesh)
    common = [
        "--resolution",
        str(args.resolution[0]),
        str(args.resolution[1]),
        "--samples",
        str(args.samples),
        "--engine",
        str(args.engine),
        "--auto_frame",
        "--auto_lights",
        "--view_transform",
        "Filmic",
        "--world_strength",
        "1.0",
        "--background_rgb",
        "1.0",
        "1.0",
        "1.0",
    ]
    if args.denoise:
        common.append("--denoise")
    common.extend(args.render_extra)

    artifacts: dict[str, str] = {}

    def _maybe_render(ply_name: str, png_name: str, extra: list[str] | None = None):
        ply_path = out_dir / ply_name
        if not ply_path.exists():
            return
        png_path = render_dir / png_name
        artifacts[png_name] = png_path.as_posix()
        render_args = list(common)
        if extra:
            render_args.extend(extra)
        _render_with_blender(args.blender, script, ply_path, png_path, render_args, dry_run=args.dry_run)

    _maybe_render("biharmonic.ply", "biharmonic.png")
    _maybe_render("biharmonic_scipy.ply", "biharmonic_scipy.png")
    _maybe_render("biharmonic_error_rel.ply", "biharmonic_error_rel.png")
    _maybe_render("biharmonic_error_logrel.ply", "biharmonic_error_logrel.png")
    _maybe_render("euclidean.ply", "euclidean.png")
    _maybe_render("geodesic.ply", "geodesic.png")

    base = out_dir / "biharmonic_scipy.ply"
    iso_gt = out_dir / "isoline_gt.ply"
    iso_pred = out_dir / "isoline_pred.ply"
    if base.exists() and iso_gt.exists() and iso_pred.exists():
        overlay_args = [
            "--base_color",
            "0.85",
            "0.85",
            "0.85",
            "--overlay",
            iso_gt.as_posix(),
            "--overlay",
            iso_pred.as_posix(),
            "--overlay_emission",
            "10.0",
        ]
        _maybe_render("biharmonic_scipy.ply", "isoline_overlay.png", extra=overlay_args)

    manifest = {
        "out_dir": out_dir.as_posix(),
        "render_dir": render_dir.as_posix(),
        "artifacts": artifacts,
    }
    (render_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

