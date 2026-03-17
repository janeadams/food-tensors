#!/usr/bin/env python3
"""Plot histogram of LLM confidence scores and write an explanatory caption."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"
FIG_DIR = ROOT / "results" / "figures"
PAPER_GENERATED = ROOT / "paper" / "sections" / "generated"
CAPTION_TEX = PAPER_GENERATED / "confidence_histogram_caption.tex"


def main() -> int:
    rows = [json.loads(line) for line in FOODS_JSONL.read_text().strip().split("\n")]
    llm_with_conf = [
        r
        for r in rows
        if r.get("source") == "llm_generated"
        and r.get("confidence") is not None
        and r.get("review_status") == "accepted"
    ]
    if not llm_with_conf:
        print("No LLM rows with confidence; nothing to plot.", file=sys.stderr)
        return 1

    df = pd.DataFrame(
        [
            {"shortname": r["shortname"], "confidence": float(r["confidence"])}
            for r in llm_with_conf
        ]
    )

    # Altair histogram: smaller figure, larger axis/tick labels
    chart = (
        alt.Chart(df)
        .mark_bar(color="#6f6f6f")
        .encode(
            alt.X("confidence:Q").bin(maxbins=15).title("Confidence"),
            alt.Y("count()").title("Count"),
        )
        .properties(width=320, height=200)
        .configure_axis(titleFontSize=16, labelFontSize=14)
        .configure_view(stroke=None)
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_html = FIG_DIR / "confidence_histogram.html"
    out_png = FIG_DIR / "confidence_histogram.png"
    chart.save(str(out_html))
    chart.save(str(out_png))
    print(f"Wrote {out_html} and {out_png}")

    # Generated caption and label for paper (full \caption and \label so \input works)
    PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    caption = (
        "Distribution of model confidence for accepted LLM-generated candidates. "
        "The shape is visibly non-parametric because these confidence values are "
        "bounded self-reports on $[0,1]$, then truncated by the acceptance cutoff "
        "at 0.75, so the observed mass piles up at a few high-confidence values "
        "rather than following a Gaussian law."
    )
    caption_tex = f"\\caption{{{caption}}}\n\\label{{fig:confidence-histogram}}\n"
    CAPTION_TEX.write_text(caption_tex)
    print(f"Wrote {CAPTION_TEX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
