#!/usr/bin/env python3
"""Build paper/main.tex into paper/main.pdf."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
MAIN_PDF = PAPER_DIR / "main.pdf"
MAIN_STEM = MAIN_TEX.stem
LATEX_AUX_SUFFIXES = (
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".lof",
    ".lot",
    ".out",
    ".run.xml",
    ".synctex.gz",
    ".toc",
)


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def cleanup_aux_files() -> None:
    """Remove LaTeX auxiliary and vestigial files from the paper directory."""
    for path in PAPER_DIR.rglob("*"):
        if path.is_file() and path.suffix in LATEX_AUX_SUFFIXES:
            path.unlink()
    for suffix in LATEX_AUX_SUFFIXES:
        path = PAPER_DIR / f"{MAIN_STEM}{suffix}"
        if path.exists():
            path.unlink()
    for match in PAPER_DIR.glob("_minted-*"):
        if match.is_dir():
            shutil.rmtree(match)


def build_with_tectonic() -> bool:
    if shutil.which("tectonic") is None:
        return False
    run(["tectonic", str(MAIN_TEX)], cwd=ROOT)
    return MAIN_PDF.exists()


def build_with_latexmk() -> bool:
    if shutil.which("latexmk") is None:
        return False
    run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ],
        cwd=PAPER_DIR,
    )
    return MAIN_PDF.exists()


def build_with_pdflatex() -> bool:
    if shutil.which("pdflatex") is None:
        return False
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER_DIR)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER_DIR)
    return MAIN_PDF.exists()


def main() -> int:
    if not MAIN_TEX.exists():
        print(f"Missing input file: {MAIN_TEX}", file=sys.stderr)
        return 1

    cleanup_aux_files()

    builders = [build_with_tectonic, build_with_latexmk, build_with_pdflatex]
    for builder in builders:
        try:
            if builder():
                cleanup_aux_files()
                print(f"Built PDF at {MAIN_PDF}")
                return 0
        except subprocess.CalledProcessError:
            continue

    print(
        "No LaTeX compiler succeeded. Install one of: tectonic, latexmk, or pdflatex.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
