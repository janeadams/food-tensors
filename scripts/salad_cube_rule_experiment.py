#!/usr/bin/env python3
"""Classify "salad" foods against Cube Rule starch-placement semantics."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from termcolor import cprint

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "experiments"
RESULTS_JSON = RESULTS_DIR / "salad_cube_rule.json"
GENERATED_TEX = ROOT / "paper" / "sections" / "generated" / "salad_cube_rule_table.tex"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TOOL_NAME = "emit_salad_classifications"

DEFAULT_SALAD_FOODS = [
    "chicken salad",
    "egg salad",
    "tuna salad",
    "ham salad",
    "potato salad",
    "macaroni salad",
    "pasta salad",
    "fruit salad",
    "broccoli salad",
    "jello salad",
    "ambrosia salad",
    "snickers salad",
    "watergate salad",
]


def log_info(message: str) -> None:
    cprint(message, "cyan")


def log_success(message: str) -> None:
    cprint(message, "green")


def log_warn(message: str) -> None:
    cprint(message, "yellow")


def tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "food_name": {"type": "string"},
                        "meets_cube_rule_salad": {"type": "boolean"},
                        "cube_type": {"type": "string"},
                        "starch_faces_present": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "structural_starch_notes": {"type": "string"},
                        "rationale_brief": {"type": "string"},
                    },
                    "required": [
                        "food_name",
                        "meets_cube_rule_salad",
                        "cube_type",
                        "starch_faces_present",
                        "structural_starch_notes",
                        "rationale_brief",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def extract_tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content", [])
    if not isinstance(content, list):
        raise ValueError("Unexpected response payload shape: missing content array")
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return tool_input
            raise ValueError("tool_use input is not an object")
    raise ValueError("No tool_use block found for structured output")


def extract_text_blocks(payload: dict[str, Any]) -> str:
    content = payload.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def coerce_results(parsed: dict[str, Any]) -> list[dict[str, Any]] | None:
    for key in ("results", "items", "classifications"):
        value = parsed.get(key)
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, dict)]
            return rows
    return None


def call_anthropic(api_key: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Return salad classifications as structured JSON matching the schema.",
                "input_schema": tool_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }
    request = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = ""
        if exc.fp is not None:
            body_text = exc.fp.read().decode("utf-8", errors="replace")
        if body_text:
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {body_text}") from exc
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {exc.reason}") from exc


def build_prompt(foods: list[str]) -> str:
    food_lines = "\n".join(f"- {food}" for food in foods)
    return f"""
You are auditing whether foods colloquially called "salad" satisfy the Cube Rule "salad" class.

Cube Rule definition to apply:
- "salad" cube_type means no structural starch faces are present on the six-face cube.
- classify by structural starch placement only.
- Ignore naming conventions, region, and whether the string includes the word "salad".

Important interpretation rule for this experiment:
- If the dish has a salad-like base plus discrete starch components mixed in (for example croutons, pasta pieces, candy/chocolate chunks, cookie crumbs, cereal, pretzel pieces), treat it as an inner-starch pattern and prefer classifying it as "nachos" rather than "salad".
- Use "salad" only when there is no meaningful structural starch component at all.

For each food:
1) decide whether it meets the Cube Rule salad definition;
2) provide the best matching cube_type label from:
   [salad, toast, sandwich, taco, sushi, quiche, calzone, cake, nachos];
3) list starch faces present using this fixed vocabulary:
   [none, top, bottom, left, right, front, back];
4) include a short structural starch note and rationale.

Foods to classify:
{food_lines}

Output only the requested tool JSON.
""".strip()


def normalize_faces(raw_faces: Any) -> list[str]:
    if not isinstance(raw_faces, list):
        return []
    normalized: list[str] = []
    allowed = {"none", "top", "bottom", "left", "right", "front", "back"}
    for face in raw_faces:
        token = str(face).strip().lower()
        if token in allowed and token not in normalized:
            normalized.append(token)
    if not normalized:
        return []
    if "none" in normalized and len(normalized) > 1:
        normalized = [face for face in normalized if face != "none"]
    return normalized


def latex_escape(value: str) -> str:
    escaped = value
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    escaped = re.sub(r"\s+", " ", escaped).strip()
    return escaped


def write_outputs(rows: list[dict[str, Any]], model: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "items": rows}
    RESULTS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    tex_lines = [
        "% Auto-generated by scripts/salad_cube_rule_experiment.py",
        "% Do not edit manually.",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\newcommand{\\tablerowrule}{\\noalign{\\color[gray]{0.75}\\hrule height 0.25pt}}",
        "\\begin{tabular}{p{0.19\\textwidth}p{0.08\\textwidth}p{0.11\\textwidth}>{\\raggedright\\arraybackslash}p{0.54\\textwidth}}",
        "\\toprule",
        "Food & Salad? & Cube Type & Rationale \\\\",
        "\\midrule",
    ]
    if rows:
        for idx, row in enumerate(rows):
            food = latex_escape(str(row.get("food_name", "")))
            meets = "yes" if bool(row.get("meets_cube_rule_salad", False)) else "no"
            if str(row.get("food_name", "")).strip().lower() == "snickers salad" and meets == "no":
                meets = r"no$^{*}$"
            cube_type = latex_escape(str(row.get("cube_type", "")))
            rationale = latex_escape(str(row.get("rationale_brief", "")))
            tex_lines.append(f"{food} & {meets} & {cube_type} & {rationale} \\\\")
            if idx != len(rows) - 1:
                tex_lines.append("\\tablerowrule")
    else:
        tex_lines.append("\\multicolumn{4}{l}{No salad-named foods were classified.} \\\\")
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Classifications for the mini experiment on foods whose names contain \\texttt{salad}. $^{*}$The narrower ontology question of whether candy-like inclusions in dishes such as Snickers salad should count as starch is discussed further in Section 6.2.}",
            "\\label{tab:salad-cube-rule}",
            "\\end{table*}",
            "",
        ]
    )
    GENERATED_TEX.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_TEX.write_text("\n".join(tex_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a salad-name experiment classifying colloquial salad foods under Cube Rule starch placement."
    )
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument(
        "--foods",
        nargs="*",
        default=DEFAULT_SALAD_FOODS,
        help="Optional override list of foods to classify (only names containing 'salad' are evaluated).",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log_warn("Missing ANTHROPIC_KEY/ANTHROPIC_API_KEY environment variable.")
        return 1

    filtered_foods = [food for food in args.foods if "salad" in food.lower()]
    if not filtered_foods:
        log_warn("No foods containing 'salad' were provided.")
        return 1

    prompt = build_prompt(filtered_foods)
    response = call_anthropic(api_key=api_key, model=args.model, prompt=prompt, max_tokens=args.max_tokens)
    try:
        parsed = extract_tool_input(response)
    except ValueError:
        text_payload = extract_text_blocks(response)
        parsed = {}
        if text_payload:
            try:
                maybe_json = json.loads(text_payload)
                if isinstance(maybe_json, dict):
                    parsed = maybe_json
            except json.JSONDecodeError:
                parsed = {}

    results = coerce_results(parsed)
    if results is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RESULTS_DIR / "salad_cube_rule_raw_response.json"
        raw_path.write_text(json.dumps(response, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        raise ValueError(f"Response missing results list; wrote raw response to {raw_path}")

    rows: list[dict[str, Any]] = []
    by_food = {food.lower(): food for food in filtered_foods}
    for item in results:
        if not isinstance(item, dict):
            continue
        food_name = str(item.get("food_name", "")).strip()
        if not food_name:
            continue
        key = food_name.lower()
        if key not in by_food:
            continue
        faces = normalize_faces(item.get("starch_faces_present", []))
        if not faces:
            faces = ["none"]
        cube_type = str(item.get("cube_type", "")).strip().lower()
        meets = bool(item.get("meets_cube_rule_salad", False))
        rows.append(
            {
                "food_name": by_food[key],
                "meets_cube_rule_salad": meets,
                "cube_type": cube_type,
                "starch_faces_present": faces,
                "structural_starch_notes": str(item.get("structural_starch_notes", "")).strip(),
                "rationale_brief": str(item.get("rationale_brief", "")).strip(),
            }
        )

    ordered_rows: list[dict[str, Any]] = []
    rows_by_name = {str(row["food_name"]).lower(): row for row in rows}
    for food in filtered_foods:
        existing = rows_by_name.get(food.lower())
        if existing is not None:
            ordered_rows.append(existing)
            continue
        ordered_rows.append(
            {
                "food_name": food,
                "meets_cube_rule_salad": False,
                "cube_type": "unknown",
                "starch_faces_present": ["none"],
                "structural_starch_notes": "No model classification returned.",
                "rationale_brief": "No model classification returned.",
            }
        )

    write_outputs(ordered_rows, model=args.model)
    yes_count = sum(1 for row in ordered_rows if bool(row["meets_cube_rule_salad"]))
    violation_count = len(ordered_rows) - yes_count
    log_success(f"Wrote experiment JSON: {RESULTS_JSON}")
    log_success(f"Wrote generated table: {GENERATED_TEX}")
    log_info(
        f"Classified {len(ordered_rows)} salad-named foods; matches={yes_count}, violations={violation_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
