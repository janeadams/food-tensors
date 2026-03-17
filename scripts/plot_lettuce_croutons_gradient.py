#!/usr/bin/env python3
"""Render the lettuce/crouton horseshoe figure used in the paper."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

# Points: (pct_lettuce, pct_croutons, label)
# (0, 100) = all croutons = salad; (100, 0) = all lettuce = salad; (50, 50) = nachos
points = pd.DataFrame(
    [
        {"pct_lettuce": 0, "pct_croutons": 100, "label": "salad"},
        {"pct_lettuce": 100, "pct_croutons": 0, "label": "salad"},
        {"pct_lettuce": 50, "pct_croutons": 50, "label": "nachos"},
    ]
)
# Dotted line: salad boundary from (0,100) to (100,0)
line_df = pd.DataFrame(
    [
        {"pct_lettuce": 0, "pct_croutons": 100},
        {"pct_lettuce": 100, "pct_croutons": 0},
    ]
)

# Black/white with gray grid lines
black = "#111111"
grid_gray = "#A0A0A0"
scale_domain = [-5, 105]
shared_x = alt.X("pct_lettuce:Q", title="% lettuce", scale=alt.Scale(domain=scale_domain))
shared_y = alt.Y("pct_croutons:Q", title="% croutons", scale=alt.Scale(domain=scale_domain))

# Dotted line (salad boundary: single-component edge)
line = (
    alt.Chart(line_df)
    .mark_line(strokeDash=[4, 2], strokeWidth=2, stroke=black)
    .encode(x=shared_x, y=shared_y)
)

# Points and text from same data
base = alt.Chart(points).encode(x=shared_x, y=shared_y)
scatter = base.mark_point(filled=True, size=120, stroke=black, strokeWidth=1.5).encode(
    color=alt.value(black)
)
annot = base.mark_text(
    align="center", baseline="middle", dx=0, dy=-28, fontSize=14, fontWeight="bold"
).encode(text="label:N", color=alt.value(black))

def save_binary_gradient() -> None:
    chart = (
        alt.layer(line, scatter, annot)
        .properties(width=400, height=400)
        .configure_axis(
            labelColor=black,
            titleColor=black,
            gridColor=grid_gray,
            titleFontSize=18,
            labelFontSize=16,
        )
        .configure_view(strokeWidth=0)
    )
    out_html = FIG_DIR / "lettuce_croutons_gradient.html"
    out_png = FIG_DIR / "lettuce_croutons_gradient.png"
    chart.save(str(out_html))
    chart.save(str(out_png))
    print(f"Saved {out_html} and {out_png}")

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_binary_gradient()
    return 0

if __name__ == "__main__":
    sys.exit(main())
