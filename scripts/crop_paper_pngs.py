#!/usr/bin/env python3
"""Crop whitespace from PNG assets referenced by paper LaTeX files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
INCLUDEGRAPHICS_PNG = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+\.png)\}")


def discover_pngs() -> list[Path]:
    files: set[Path] = set()
    for tex_path in PAPER_DIR.rglob("*.tex"):
        text = tex_path.read_text(encoding="utf-8")
        for rel in INCLUDEGRAPHICS_PNG.findall(text):
            candidates = [
                (tex_path.parent / rel).resolve(),
                (PAPER_DIR / rel).resolve(),
            ]
            for candidate in candidates:
                if candidate.exists():
                    files.add(candidate)
                    break
    return sorted(files)


def crop_png(path: Path, padding: int) -> bool:
    rgba = Image.open(path).convert("RGBA")
    alpha_bbox = rgba.getchannel("A").getbbox()

    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb_on_white = Image.alpha_composite(white_bg, rgba).convert("RGB")
    white_rgb = Image.new("RGB", rgb_on_white.size, (255, 255, 255))
    rgb_bbox = ImageChops.difference(rgb_on_white, white_rgb).getbbox()

    bboxes = [bbox for bbox in (alpha_bbox, rgb_bbox) if bbox is not None]
    if not bboxes:
        return False

    left = max(0, min(b[0] for b in bboxes) - padding)
    upper = max(0, min(b[1] for b in bboxes) - padding)
    right = min(rgba.width, max(b[2] for b in bboxes) + padding)
    lower = min(rgba.height, max(b[3] for b in bboxes) + padding)

    cropped = rgba.crop((left, upper, right, lower))
    cropped.save(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop whitespace around PNGs used in the paper.")
    parser.add_argument("--padding", type=int, default=8, help="Extra pixels retained around cropped content")
    args = parser.parse_args()

    pngs = discover_pngs()
    if not pngs:
        print("No PNG files referenced by LaTeX were found.")
        return 0

    changed = 0
    for png in pngs:
        if crop_png(png, padding=args.padding):
            changed += 1
            print(f"Cropped: {png}")
        else:
            print(f"Unchanged: {png}")
    print(f"Cropped {changed} of {len(pngs)} PNG files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
