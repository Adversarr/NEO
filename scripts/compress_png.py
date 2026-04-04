"""Compress a PNG (or other image) to a compressed format such as jpg/jpeg."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _compress_image(
    input_path: Path,
    output_path: Path,
    quality: int,
    fmt: str | None = None,
) -> None:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input image not found: {src}")

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    fmt_lower = fmt.lower() if fmt is not None else ""
    if not fmt_lower:
        fmt_lower = dst.suffix.lstrip(".").lower()
    if not fmt_lower:
        fmt_lower = "jpg"

    q = int(max(1, min(100, int(quality))))

    pil_format = fmt_lower
    if fmt_lower in {"jpg", "jpeg"}:
        pil_format = "JPEG"
    elif fmt_lower == "tif":
        pil_format = "TIFF"

    with Image.open(src.as_posix()) as im:
        out_im = im
        if fmt_lower in {"jpg", "jpeg"} and out_im.mode in {"RGBA", "LA", "P"}:
            rgba = out_im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            out_im = Image.alpha_composite(bg, rgba).convert("RGB")
        elif fmt_lower in {"jpg", "jpeg"} and out_im.mode != "RGB":
            out_im = out_im.convert("RGB")

        save_kwargs: dict = {}
        if fmt_lower in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": q, "optimize": True, "progressive": True})
        elif fmt_lower in {"webp"}:
            save_kwargs.update({"quality": q, "method": 6})
        else:
            save_kwargs.update({"quality": q})

        out_im.save(dst.as_posix(), format=pil_format, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--format", type=str, default="")
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = args.format.lstrip(".") if args.format else "jpg"
        output_path = input_path.with_suffix("." + suffix)

    fmt = args.format or ""
    _compress_image(input_path, output_path, quality=int(args.quality), fmt=fmt or None)


if __name__ == "__main__":
    main()
