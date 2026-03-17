#!/usr/bin/env python3
"""Generate tensor visualization outputs from real-only accepted foods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from termcolor import cprint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensor_food.io_utils import read_jsonl
from tensor_food.schema import CUBE_TYPES, PROTEIN_TYPES, STARCH_TYPES, is_structurally_invalid_cell

FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"
FIG_DIR = ROOT / "results" / "figures"
PAPER_GENERATED_DIR = ROOT / "paper" / "sections" / "generated"
CUBE_LABELS = {idx: human for idx, human in enumerate([t.replace("_", " ").title() for t in CUBE_TYPES])}
PROTEIN_LABELS = {
    idx: human for idx, human in enumerate([t.replace("_", " ").title() for t in PROTEIN_TYPES])
}
STARCH_LABELS = {
    idx: human for idx, human in enumerate([t.replace("_", " ").title() for t in STARCH_TYPES])
}

CLASS_CANONICAL = "Cube Rule canonical example"
CLASS_IDENTIFIED = "LLM identified real example"
CLASS_SPECULATED = "LLM generated speculative example"


def humanize(label: str) -> str:
    return label.replace("_", " ").title()


def idx_to_label_map(index_col: str) -> dict[int, str]:
    if index_col == "cube_idx":
        return CUBE_LABELS
    if index_col == "protein_idx":
        return PROTEIN_LABELS
    if index_col == "starch_idx":
        return STARCH_LABELS
    raise ValueError(f"Unsupported index column: {index_col}")


def dendrogram_order(
    profiles: pd.DataFrame,
    category_col: str,
    value_col: str = "occupied",
    fallback_order: list[int] | None = None,
) -> list[int]:
    if profiles.empty:
        return fallback_order or []
    matrix = (
        profiles.pivot_table(
            index=category_col,
            columns=[col for col in profiles.columns if col not in {category_col, value_col}],
            values=value_col,
            fill_value=0,
            aggfunc="max",
        )
        .sort_index()
        .astype(float)
    )
    categories = matrix.index.to_list()
    if len(categories) < 2:
        return categories
    try:
        distances = pdist(matrix.values, metric="jaccard")
        if np.allclose(distances, 0):
            return categories
        tree = linkage(distances, method="average")
        tree = optimal_leaf_ordering(tree, distances)
        ordered = [categories[i] for i in leaves_list(tree)]
        return ordered
    except Exception:
        return fallback_order or categories


def pair_projection(df: pd.DataFrame, x: str, y: str, output: Path, title: str) -> dict[str, object]:
    classified = (
        df.assign(
            cell_class=lambda d: np.select(
                [
                    d["source"].eq("cube_rule_prefill"),
                    d["source"].eq("llm_generated") & d["review_status"].eq("accepted") & d["is_real"].eq(True),
                    d["source"].eq("llm_generated") & d["review_status"].ne("rejected") & d["is_real"].eq(False),
                ],
                [CLASS_CANONICAL, CLASS_IDENTIFIED, CLASS_SPECULATED],
                default="other",
            )
        )
    )
    occupied = (
        classified.groupby([x, y], as_index=False)
        .agg(
            count=("food_id", "count"),
            sample_foods=("shortname", lambda s: ", ".join(sorted(set(s))[:5])),
            has_canonical=("cell_class", lambda s: (s == CLASS_CANONICAL).any()),
            has_identified=("cell_class", lambda s: (s == CLASS_IDENTIFIED).any()),
            has_speculated=("cell_class", lambda s: (s == CLASS_SPECULATED).any()),
        )
        .assign(
            cell_class=lambda d: np.select(
                [d["has_canonical"], d["has_identified"], d["has_speculated"]],
                [CLASS_CANONICAL, CLASS_IDENTIFIED, CLASS_SPECULATED],
                default="empty",
            )
        )
    )
    x_active = sorted(occupied[x].unique().tolist())
    y_active = sorted(occupied[y].unique().tolist())
    if not x_active or not y_active:
        cprint(f"Skipping {title}: no active rows/columns for {x}/{y}", "yellow")
        return {"x_axis": humanize(x.replace("_idx", "")), "y_axis": humanize(y.replace("_idx", "")), "omitted_x": [], "omitted_y": []}

    grid = (
        pd.MultiIndex.from_product([x_active, y_active], names=[x, y]).to_frame(index=False)
    )
    x_labels = idx_to_label_map(x)
    y_labels = idx_to_label_map(y)
    occupancy_profiles = (
        grid.merge(occupied[[x, y, "count"]], on=[x, y], how="left")
        .assign(occupied=lambda d: d["count"].fillna(0).astype(int) > 0)
        [[x, y, "occupied"]]
    )
    x_order_idx = dendrogram_order(
        occupancy_profiles.rename(columns={x: "category", y: "other_axis"}),
        category_col="category",
        fallback_order=x_active,
    )
    y_order_idx = dendrogram_order(
        occupancy_profiles.rename(columns={y: "category", x: "other_axis"}),
        category_col="category",
        fallback_order=y_active,
    )
    x_order = [x_labels[idx] for idx in x_order_idx]
    y_order = [y_labels[idx] for idx in y_order_idx]
    omitted_x = [label for idx, label in x_labels.items() if idx not in x_active]
    omitted_y = [label for idx, label in y_labels.items() if idx not in y_active]

    plot_df = (
        grid.merge(occupied, on=[x, y], how="left")
        .assign(
            count=lambda d: d["count"].fillna(0).astype(int),
            sample_foods=lambda d: d["sample_foods"].fillna(""),
            cell_class=lambda d: d["cell_class"].fillna("empty"),
        )
        .assign(occupied=lambda d: d["count"] > 0)
        .assign(
            x_label=lambda d: d[x].map(x_labels),
            y_label=lambda d: d[y].map(y_labels),
        )
    )
    x_title = humanize(x.replace("_idx", ""))
    y_title = humanize(y.replace("_idx", ""))

    base = (
        alt.Chart(plot_df)
        .transform_filter("datum.cell_class != 'empty'")
        .encode(
            x=alt.X(
                "x_label:N",
                sort=x_order,
                title=x_title,
                axis=alt.Axis(labelAngle=0, labelFontSize=20, titleFontSize=24),
            ),
            y=alt.Y(
                "y_label:N",
                sort=y_order,
                title=y_title,
                axis=alt.Axis(labelFontSize=20, titleFontSize=24),
            ),
            size=alt.value(280),
            tooltip=["x_label", "y_label", "cell_class", "count", "sample_foods"],
        )
    )
    layer_canonical = base.transform_filter(
        f"datum.cell_class == '{CLASS_CANONICAL}'"
    ).mark_square(color="#000000", stroke="#000000", strokeWidth=0.6)
    layer_identified = base.transform_filter(
        f"datum.cell_class == '{CLASS_IDENTIFIED}'"
    ).mark_square(color="#8a8a8a", stroke="#8a8a8a", strokeWidth=0.6)
    layer_speculated = base.transform_filter(
        f"datum.cell_class == '{CLASS_SPECULATED}'"
    ).mark_square(color="#ffffff", stroke="#000000", strokeWidth=2.0)
    chart = alt.layer(layer_canonical, layer_identified, layer_speculated).properties(
        width=720, height=520
    )
    chart.save(str(output))
    chart.save(str(output.with_suffix(".png")))
    cprint(f"Wrote projection: {output}", "green")
    return {
        "x_axis": x_title,
        "y_axis": y_title,
        "omitted_x": omitted_x,
        "omitted_y": omitted_y,
    }


def canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["source"] == "cube_rule_prefill")
        & (df["review_status"] != "rejected")
    ].copy()


def llm_identified_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["source"] == "llm_generated")
        & (df["review_status"] == "accepted")
        & (df["is_real"] == True)
    ].copy()


def llm_speculated_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["source"] == "llm_generated")
        & (df["review_status"] != "rejected")
        & (df["is_real"] == False)
    ].copy()


def render_stage(
    stage_df: pd.DataFrame,
    title_prefix: str,
    slug: str,
) -> dict[str, dict[str, object]]:
    if stage_df.empty:
        cprint(f"Skipping stage '{slug}': no rows", "yellow")
        return {}
    stage_dir = FIG_DIR / slug
    stage_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, str, Path, str, str]] = [
        (
            "cube_idx",
            "protein_idx",
            stage_dir / "tensor_cube_vs_protein.html",
            f"{title_prefix}: Cube Type vs. Protein",
            "cube_vs_protein",
        ),
        (
            "cube_idx",
            "starch_idx",
            stage_dir / "tensor_cube_vs_starch.html",
            f"{title_prefix}: Cube Type vs. Starch",
            "cube_vs_starch",
        ),
        (
            "protein_idx",
            "starch_idx",
            stage_dir / "tensor_protein_vs_starch.html",
            f"{title_prefix}: Protein vs. Starch",
            "protein_vs_starch",
        ),
    ]
    omission_meta: dict[str, dict[str, object]] = {}
    for x, y, output, title, key in tqdm(outputs, desc=f"Render stage {slug}", unit="chart", leave=False):
        omission_meta[key] = pair_projection(stage_df, x=x, y=y, output=output, title=title)
    (stage_dir / "omitted_axes.json").write_text(
        json.dumps(omission_meta, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    cprint(f"Wrote omission metadata: {stage_dir / 'omitted_axes.json'}", "green")
    return omission_meta


def format_omitted(items: list[str]) -> str:
    return "none" if not items else ", ".join(items)


def compact_projection_omission(
    proj_label: str,
    x_axis: str,
    y_axis: str,
    omitted_x: str,
    omitted_y: str,
) -> str:
    return f"{proj_label} ({x_axis}: {omitted_x}; {y_axis}: {omitted_y})"


def write_omission_tex(stage_omissions: dict[str, dict[str, dict[str, object]]]) -> None:
    stage_labels = {
        "canonical": "Canonical",
        "canonical_plus_llm_identified": "Identified",
        "canonical_plus_llm_identified_plus_llm_speculated": "Speculated",
    }
    projection_labels = {
        "cube_vs_protein": "Cube vs. Protein",
        "cube_vs_starch": "Cube vs. Starch",
        "protein_vs_starch": "Protein vs. Starch",
    }
    lines = [
        "% Auto-generated by scripts/plot_tensor.py",
        "% Do not edit manually.",
    ]
    for slug, stage_key in stage_labels.items():
        projections = stage_omissions.get(slug, {})
        summary_parts: list[str] = []
        for proj_key, proj_label in projection_labels.items():
            info = projections.get(proj_key, {})
            x_axis = str(info.get("x_axis", "X"))
            y_axis = str(info.get("y_axis", "Y"))
            omitted_x = format_omitted(list(info.get("omitted_x", [])))
            omitted_y = format_omitted(list(info.get("omitted_y", [])))
            summary_parts.append(
                compact_projection_omission(
                    proj_label=proj_label,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    omitted_x=omitted_x,
                    omitted_y=omitted_y,
                )
            )
        summary = " Omitted categories by projection: " + "; ".join(summary_parts) + "."
        lines.append(f"\\newcommand{{\\omit{stage_key}Summary}}{{{summary}}}")
    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = PAPER_GENERATED_DIR / "figure_omissions.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cprint(f"Wrote generated LaTeX omissions: {tex_path}", "green")


def render_stage_3d(stage_df: pd.DataFrame, title_prefix: str, slug: str, export_png: bool) -> None:
    if stage_df.empty:
        cprint(f"Skipping 3D stage '{slug}': no rows", "yellow")
        return
    stage_dir = FIG_DIR / slug
    stage_dir.mkdir(parents=True, exist_ok=True)

    df = stage_df
    active_cube_idx = sorted(df["cube_idx"].unique().tolist())
    active_protein_idx = sorted(df["protein_idx"].unique().tolist())
    active_starch_idx = sorted(df["starch_idx"].unique().tolist())
    grid_3d = pd.MultiIndex.from_product(
        [active_cube_idx, active_protein_idx, active_starch_idx],
        names=["cube_idx", "protein_idx", "starch_idx"],
    ).to_frame(index=False)
    classified_3d = df.assign(
        cell_class=lambda d: np.select(
            [
                d["source"].eq("cube_rule_prefill"),
                d["source"].eq("llm_generated") & d["review_status"].eq("accepted") & d["is_real"].eq(True),
                d["source"].eq("llm_generated") & d["review_status"].ne("rejected") & d["is_real"].eq(False),
            ],
            [CLASS_CANONICAL, CLASS_IDENTIFIED, CLASS_SPECULATED],
            default="other",
        )
    )
    occupied_3d = (
        classified_3d.groupby(["cube_idx", "protein_idx", "starch_idx"], as_index=False)
        .agg(
            count=("food_id", "count"),
            sample_foods=("shortname", lambda s: ", ".join(sorted(set(s))[:5])),
            has_canonical=("cell_class", lambda s: (s == CLASS_CANONICAL).any()),
            has_identified=("cell_class", lambda s: (s == CLASS_IDENTIFIED).any()),
            has_speculated=("cell_class", lambda s: (s == CLASS_SPECULATED).any()),
        )
        .assign(occupied=True)
        .assign(
            cell_class=lambda d: np.select(
                [d["has_canonical"], d["has_identified"], d["has_speculated"]],
                [CLASS_CANONICAL, CLASS_IDENTIFIED, CLASS_SPECULATED],
                default=CLASS_IDENTIFIED,
            )
        )
    )
    occupancy_3d = (
        grid_3d.merge(
            occupied_3d[["cube_idx", "protein_idx", "starch_idx", "occupied"]],
            on=["cube_idx", "protein_idx", "starch_idx"],
            how="left",
        )
        .assign(occupied=lambda d: d["occupied"].fillna(False))
    )
    cube_order_idx = dendrogram_order(
        occupancy_3d.rename(
            columns={
                "cube_idx": "category",
                "protein_idx": "other_a",
                "starch_idx": "other_b",
            }
        )[["category", "other_a", "other_b", "occupied"]],
        category_col="category",
        fallback_order=active_cube_idx,
    )
    protein_order_idx = dendrogram_order(
        occupancy_3d.rename(
            columns={
                "protein_idx": "category",
                "cube_idx": "other_a",
                "starch_idx": "other_b",
            }
        )[["category", "other_a", "other_b", "occupied"]],
        category_col="category",
        fallback_order=active_protein_idx,
    )
    starch_order_idx = dendrogram_order(
        occupancy_3d.rename(
            columns={
                "starch_idx": "category",
                "cube_idx": "other_a",
                "protein_idx": "other_b",
            }
        )[["category", "other_a", "other_b", "occupied"]],
        category_col="category",
        fallback_order=active_starch_idx,
    )
    cube_order_labels = [CUBE_LABELS[idx] for idx in cube_order_idx]
    protein_order_labels = [PROTEIN_LABELS[idx] for idx in protein_order_idx]
    starch_order_labels = [STARCH_LABELS[idx] for idx in starch_order_idx]
    plot_3d = occupied_3d.assign(
        cube_type=lambda d: d["cube_idx"].map(dict(enumerate(CUBE_TYPES))).map(humanize),
        protein_type=lambda d: d["protein_idx"].map(dict(enumerate(PROTEIN_TYPES))).map(humanize),
        starch_type=lambda d: d["starch_idx"].map(dict(enumerate(STARCH_TYPES))).map(humanize),
    )
    fig = px.scatter_3d(
        plot_3d,
        x="cube_type",
        y="protein_type",
        z="starch_type",
        color="cell_class",
        symbol="cell_class",
        color_discrete_map={
            CLASS_CANONICAL: "#000000",
            CLASS_IDENTIFIED: "#8a8a8a",
            CLASS_SPECULATED: "#ffffff",
        },
        symbol_map={
            CLASS_CANONICAL: "square",
            CLASS_IDENTIFIED: "square",
            CLASS_SPECULATED: "square",
        },
        hover_data={
            "cell_class": True,
            "count": True,
            "sample_foods": True,
        },
    )
    fig.update_traces(
        marker=dict(size=5, opacity=1.0, line=dict(width=0.2, color="#222222")),
        selector=dict(name=CLASS_CANONICAL),
    )
    fig.update_traces(
        marker=dict(size=5, opacity=1.0, line=dict(width=0.2, color="#8a8a8a")),
        selector=dict(name=CLASS_IDENTIFIED),
    )
    fig.update_traces(
        marker=dict(size=5, opacity=1.0, line=dict(width=2.0, color="#000000")),
        selector=dict(name=CLASS_SPECULATED),
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        paper_bgcolor="white",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(
                categoryorder="array",
                categoryarray=cube_order_labels,
                tickfont=dict(size=18),
                title=dict(text=""),
            ),
            yaxis=dict(
                categoryorder="array",
                categoryarray=protein_order_labels,
                tickfont=dict(size=18),
                title=dict(text=""),
            ),
            zaxis=dict(
                categoryorder="array",
                categoryarray=starch_order_labels,
                tickfont=dict(size=18),
                title=dict(text=""),
            ),
        ),
        scene_camera=dict(eye=dict(x=1.35, y=1.25, z=0.95)),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    output_html = stage_dir / "tensor_3d.html"
    fig.write_html(str(output_html))

    if export_png:
        for static_path in (stage_dir / "tensor_3d.png", stage_dir / "tensor_3d.pdf"):
            try:
                fig.write_image(str(static_path), width=2000, height=1400, scale=2)
                cprint(f"Wrote static 3D snapshot to {static_path}", "green")
            except Exception as exc:
                cprint(f"Could not export {static_path.suffix.upper()} ({exc}).", "yellow")
                cprint("Install Chrome and retry: uv run plotly_get_chrome", "yellow")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tensor visualization artifacts.")
    parser.add_argument(
        "--include-3d",
        action="store_true",
        help="Also render optional Plotly 3D scatter output",
    )
    parser.add_argument(
        "--export-3d-png",
        action="store_true",
        help="Also export a static PNG snapshot of the 3D plot for paper figures",
    )
    args = parser.parse_args()

    rows = read_jsonl(FOODS_JSONL)
    if not rows:
        cprint(f"No rows in {FOODS_JSONL}. Run prefill/review/build steps first.", "yellow")
        return 1
    df = pd.DataFrame(rows)
    # Exclude rows whose cube morphology and starch axis disagree.
    df = df[
        ~df.apply(
            lambda r: is_structurally_invalid_cell(
                r["cube_type"], r["protein_type"], r["starch_type"]
            ),
            axis=1,
        )
    ]
    cprint(f"Loaded {len(df)} rows from {FOODS_JSONL} (structurally invalid cells excluded)", "cyan")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stage_canonical = canonical_rows(df)
    stage_llm_identified = llm_identified_rows(df)
    stage_llm_speculated = llm_speculated_rows(df)

    stage_canonical_plus_identified = pd.concat(
        [stage_canonical, stage_llm_identified], ignore_index=True
    ).drop_duplicates(subset=["food_id"])
    stage_all = pd.concat(
        [stage_canonical_plus_identified, stage_llm_speculated], ignore_index=True
    ).drop_duplicates(subset=["food_id"])

    stage_configs = [
        (stage_canonical, "Canonical", "canonical"),
        (stage_canonical_plus_identified, "Canonical + LLM Identified", "canonical_plus_llm_identified"),
        (
            stage_all,
            "Canonical + LLM Speculated",
            "canonical_plus_llm_identified_plus_llm_speculated",
        ),
    ]
    stage_omissions: dict[str, dict[str, dict[str, object]]] = {}
    for stage_df, title_prefix, slug in tqdm(stage_configs, desc="2D stage groups", unit="stage"):
        stage_omissions[slug] = render_stage(stage_df, title_prefix=title_prefix, slug=slug)
    write_omission_tex(stage_omissions)

    if args.include_3d:
        stage_3d_configs = [
            (
                stage_canonical,
                "Tensor Occupancy (Canonical Cube Rule Examples)",
                "canonical",
            ),
            (
                stage_canonical_plus_identified,
                "Tensor Occupancy (Canonical + LLM Identified Real Foods)",
                "canonical_plus_llm_identified",
            ),
            (
                stage_all,
                "Tensor Occupancy (Canonical + LLM Identified + LLM Speculated)",
                "canonical_plus_llm_identified_plus_llm_speculated",
            ),
        ]
        for stage_df, title_prefix, slug in tqdm(stage_3d_configs, desc="3D stage groups", unit="stage"):
            render_stage_3d(stage_df, title_prefix=title_prefix, slug=slug, export_png=args.export_3d_png)

    cprint(f"Wrote visualization outputs to {FIG_DIR}", "green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
