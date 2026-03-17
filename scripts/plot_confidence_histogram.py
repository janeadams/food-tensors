#!/usr/bin/env python3
"""Plot histogram of LLM confidence scores and write caption with top/bottom examples."""

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

    # Top 3 and bottom 3 by confidence (tie-break by shortname for determinism)
    sorted_asc = df.sort_values(["confidence", "shortname"])
    sorted_desc = df.sort_values(["confidence", "shortname"], ascending=[False, True])
    lowest = sorted_asc.head(3)
    highest = sorted_desc.head(3)

    def fmt_examples(sub: pd.DataFrame) -> str:
        return ", ".join(
            f"{row['shortname']} ({row['confidence']:.2f})"
            for _, row in sub.iterrows()
        )

    highest_str = fmt_examples(highest)
    lowest_str = fmt_examples(lowest)

    # Altair histogram: smaller figure, larger axis/tick labels
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("confidence:Q").bin(maxbins=15).title("Confidence"),
            alt.Y("count()").title("Count"),
        )
        .properties(width=320, height=200, title="LLM confidence (accepted candidates)")
        .configure_axis(titleFontSize=16, labelFontSize=14)
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
        f"Highest confidence: {highest_str}. "
        f"Lowest confidence: {lowest_str}."
    )
    caption_tex = f"\\caption{{{caption}}}\n\\label{{fig:confidence-histogram}}\n"
    CAPTION_TEX.write_text(caption_tex)
    print(f"Wrote {CAPTION_TEX}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
