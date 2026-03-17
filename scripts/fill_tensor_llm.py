#!/usr/bin/env python3
"""Fill empty tensor cells with Anthropic-generated candidate foods."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from termcolor import cprint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensor_food.io_utils import ensure_parent, read_jsonl, write_jsonl
from tensor_food.schema import (
    CUBE_TYPES,
    PROTEIN_TYPES,
    STARCH_TYPES,
    FoodRecord,
    is_salad_starch_prohibited,
    normalize_food_id,
    validate_axes,
)

FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"
BATCH_DIR = ROOT / "results" / "llm_batches"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _api_key_from_zshrc(*names: str) -> str | None:
    """If env is missing, try to read export VAR=value from ~/.zshrc (no shell eval)."""
    zshrc = Path.home() / ".zshrc"
    if not zshrc.exists():
        return None
    pattern = re.compile(r"^\s*export\s+(" + "|".join(re.escape(n) for n in names) + r")\s*=\s*(.+)\s*$")
    for line in zshrc.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line.strip())
        if m:
            value = m.group(2).strip().strip("'\"")
            if value:
                return value
    return None

CUBE_MORPHOLOGY = {
    "salad": {
        "morphology_name": "no_structural_starch",
        "definition": "No structural starch faces are present on the 6-face cube.",
        "examples": [
            "grilled steak",
            "mashed potatoes",
            "tomato soup (wet salad)",
        ],
    },
    "toast": {
        "morphology_name": "bottom_face_only",
        "definition": "Exactly one structural starch face: bottom only.",
        "examples": [
            "pizza slice",
            "nigiri sushi",
            "pumpkin pie slice",
        ],
    },
    "sandwich": {
        "morphology_name": "top_and_bottom_faces",
        "definition": "Two structural starch faces: top and bottom.",
        "examples": [
            "quesadilla (non-folded)",
            "toast sandwich",
            "victoria sponge",
        ],
    },
    "taco": {
        "morphology_name": "left_bottom_right_faces",
        "definition": "Three structural starch faces: left, bottom, and right.",
        "examples": [
            "hot dog",
            "sub sandwich (uncut)",
            "slice of pie (on its side)",
        ],
    },
    "sushi": {
        "morphology_name": "left_right_bottom_top_faces",
        "definition": "Four structural starch faces: left, right, bottom, and top.",
        "examples": [
            "falafel wrap",
            "pigs in a blanket",
            "enchilada",
        ],
    },
    "quiche": {
        "morphology_name": "five_faces_enclosed_open_top",
        "definition": "Five structural starch faces: bottom and all four sides (open top).",
        "examples": [
            "soup in bread bowl",
            "deep-dish pizza",
            "key lime pie",
        ],
    },
    "calzone": {
        "morphology_name": "all_six_faces_enclosed",
        "definition": "All six structural starch faces are present (fully enclosed).",
        "examples": [
            "burrito",
            "corn dog",
            "dumpling",
            "pop-tart",
        ],
    },
    "cake": {
        "morphology_name": "stacked_layers",
        "definition": "Three stacked starch layers in vertical arrangement.",
        "examples": [
            "lasagna",
            "big mac",
            "stack of flapjacks",
        ],
    },
    "nachos": {
        "morphology_name": "outer_container_plus_inner_starch",
        "definition": "A starch structure with an additional smaller starch component inside.",
        "examples": [
            "poutine",
            "cereal in milk",
            "ramen (wet nachos)",
        ],
    },
}

STARCH_REQUIRED_TERMS = {
    "wheat": ["bread", "bun", "pita", "tortilla", "wheat", "flatbread", "roll", "crust"],
    "rice": ["rice", "risotto", "arborio", "sushi rice"],
    "corn": ["corn", "masa", "arepa", "polenta", "cornmeal"],
    "potato": ["potato", "tater", "hash brown", "gnocchi"],
    "pasta_noodle": ["pasta", "noodle", "spaghetti", "ramen", "macaroni", "orzo"],
    "pastry_dough": ["pastry", "puff", "phyllo", "pie crust", "tart", "dough"],
    "other_grain": ["barley", "quinoa", "couscous", "oat", "grain"],
}

STARCH_FORBIDDEN_TERMS = {
    # User-requested strictness: pastry_dough cannot silently map from pita/bread.
    "pastry_dough": ["pita", "bread", "bun", "tortilla", "roll", "flatbread"],
}

PROTEIN_FORBIDDEN_WHEN_NONE = [
    "beef",
    "pork",
    "chicken",
    "turkey",
    "fish",
    "shrimp",
    "tuna",
    "salmon",
    "egg",
    "tofu",
    "bean",
    "lentil",
    "cheese",
    "milk",
    "cream",
    "yogurt",
    "whey",
    "casein",
    "butter",
]

PROTEIN_REQUIRED_TERMS = {
    "red_meat": ["beef", "pork", "lamb", "sausage", "bacon"],
    "poultry": ["chicken", "turkey", "duck"],
    "seafood": ["fish", "shrimp", "tuna", "salmon", "haddock", "seafood"],
    "egg": ["egg", "omelet", "frittata"],
    "plant_protein": ["tofu", "bean", "lentil", "chickpea", "tempeh", "falafel"],
    "dairy": ["cheese", "paneer", "dairy", "mozzarella", "milk", "cream", "yogurt"],
}

DISALLOWED_COMPOSITION_PHRASES = [
    "served with",
    "with a side of",
    "alongside",
    "accompanied by",
    "paired with",
    "over ",
]

PROTEIN_CATEGORY_TERMS = {
    "red_meat": ["beef", "pork", "lamb", "sausage", "bacon"],
    "poultry": ["chicken", "turkey", "duck"],
    "seafood": ["fish", "shrimp", "tuna", "salmon", "haddock", "seafood"],
    "egg": ["egg", "omelet", "frittata"],
    "plant_protein": ["tofu", "bean", "lentil", "chickpea", "tempeh", "falafel"],
    "dairy": ["cheese", "paneer", "dairy", "mozzarella", "milk", "cream", "yogurt"],
}

# Cube Rule canonical morphology: dish terms that belong to a specific cube_type only.
# Reject candidates that mention these when the requested cube_type does not match.
CANONICAL_MORPHOLOGY_TERMS: dict[str, str] = {
    "ramen": "nachos",  # ramen = wet nachos (outer container + inner starch), not salad
    "poutine": "nachos",
}

T = TypeVar("T")
TOOL_NAME = "emit_food_candidates"


def log_info(message: str) -> None:
    cprint(message, "cyan")


def log_success(message: str) -> None:
    cprint(message, "green")


def log_warn(message: str) -> None:
    cprint(message, "yellow")


def candidate_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "shortname": {"type": "string"},
                        "description": {"type": "string"},
                        "is_real": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale_brief": {"type": "string"},
                    },
                    "required": [
                        "shortname",
                        "description",
                        "is_real",
                        "confidence",
                        "rationale_brief",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


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


def call_anthropic(api_key: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "name": TOOL_NAME,
                "description": "Return candidate foods as structured JSON matching the schema.",
                "input_schema": candidate_tool_schema(),
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
        body = ""
        if exc.fp is not None:
            body = exc.fp.read().decode("utf-8", errors="replace")
        if body:
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {body}") from exc
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {exc.reason}") from exc


def build_prompt(cube_type: str, protein_type: str, starch_type: str, candidates_per_cell: int) -> str:
    target = CUBE_MORPHOLOGY[cube_type]
    examples = "; ".join(target["examples"])
    crash_course = "\n".join(
        f"- {name} [{meta['morphology_name']}]: {meta['definition']}"
        for name, meta in CUBE_MORPHOLOGY.items()
    )
    return f"""
You generate candidate foods for a strict food morphology tensor.

Cube Rule crash course (morphology-first, not casual naming):
{crash_course}

Important edge-case reminders:
- Nigiri sushi maps to toast because structural starch is bottom-only.
- Ramen is a Cube Rule example of NACHOS (outer container + inner starch), not salad. Do not assign ramen or ramen-broth dishes to salad morphology.
- Use structural starch placement, not everyday naming.

Category semantics are morphology-only:
- "salad" means no structural starch faces; it does NOT imply cold temperature.
- "taco" means left-bottom-right starch placement; it does NOT imply spice level or cuisine.
- "sushi" means four-face enclosure pattern; it does NOT imply Japanese ingredients.
- "calzone" means full six-face enclosure; it does NOT imply Italian flavor profile.
- Ignore flavor descriptors (spicy/sweet), serving temperature (hot/cold), and cuisine labels unless they affect starch placement.

Axis definitions:
- cube_type categories: {CUBE_TYPES}
- protein_type categories: {PROTEIN_TYPES}
- starch_type categories: {STARCH_TYPES}

Protein class semantics:
- protein_type=none means no meaningful protein category contribution.
- Dairy ingredients (e.g., milk, cream, yogurt, cheese) count as dairy protein.
- Legumes/soy/tofu count as plant_protein.
- Eggs count as egg.
- If a dish has a clear protein source, do not label it as protein_type=none.

Target tensor cell:
- cube_type={cube_type}
- cube morphology alias={target['morphology_name']}
- cube morphology definition={target['definition']}
- protein_type={protein_type}
- starch_type={starch_type}

Canonical examples for this cube_type from Cube Rule:
- {examples}

Task:
- Return up to {candidates_per_cell} candidate foods for this exact cell.
- Prefer known real foods; if none plausible, return an empty candidates array.
- Conservative behavior is required: if the answer is not obvious, return an empty candidates array.
- Keep shortname concise.
- Keep description to one sentence.
- Reject candidates that do not match the exact cube morphology definition above.
- If uncertain about morphology fit, do not guess; return empty list.
- HARD CONSTRAINT: match the requested starch_type exactly; do not substitute related starches.
- HARD CONSTRAINT: match the requested protein_type exactly; do not substitute neighboring protein categories.
- Example strictness: if starch_type is pastry_dough, do not return pita/bread/tortilla foods.
- HARD CONSTRAINT: return a single canonical dish/entity, not a composed plate with separate components.
- Reject formats like "X served with Y", "X alongside Y", or "X over Y" when X and Y are separate items.
- HARD CONSTRAINT: do not invent weak matches to fill the cell; empty output is valid and preferred when uncertain.
- Prefer globally recognized, canonical dish names over ad-hoc compositional labels.
- If only a generic descriptive label is available (e.g., "beef pasta salad"), return empty list instead of forcing a weak candidate.

Good atomic examples:
- "beef congee" (single dish)
- "corned beef hash" (single dish)
- "spaghetti and meatballs" (canonical, widely recognized)

Bad non-atomic examples:
- "grilled steak served with corn salsa"
- "steak alongside roasted potatoes"
- "beef pasta salad" (generic compositional label unless strongly established as a standard dish name)

Output JSON only, no markdown, with this exact shape:
{{
  "candidates": [
    {{
      "shortname": "string",
      "description": "string",
      "is_real": true,
      "confidence": 0.0,
      "rationale_brief": "must explain morphology fit in one short sentence"
    }}
  ]
}}
""".strip()


def build_prompt_speculative(cube_type: str, protein_type: str, starch_type: str, candidates_per_cell: int) -> str:
    """Prompt for novel/plausible food ideas (is_real=false) under same strict morphology/starch/protein rules."""
    target = CUBE_MORPHOLOGY[cube_type]
    examples = "; ".join(target["examples"])
    crash_course = "\n".join(
        f"- {name} [{meta['morphology_name']}]: {meta['definition']}"
        for name, meta in CUBE_MORPHOLOGY.items()
    )
    return f"""
You propose novel or plausible food ideas for a strict food morphology tensor.
Same axis and constraint rules apply as for real foods, but you may invent new dishes or name obscure ones.

Cube Rule crash course (morphology-first):
{crash_course}

Canonical morphology: ramen is NACHOS (outer container + inner starch), not salad. Do not propose ramen or ramen-broth dishes for salad cells.

Category semantics are morphology-only (e.g. "salad" = no structural starch; "taco" = left-bottom-right starch; no temperature/cuisine requirements).

Axis definitions:
- cube_type: {CUBE_TYPES}
- protein_type: {PROTEIN_TYPES}
- starch_type: {STARCH_TYPES}

Protein: none = no meaningful protein; dairy/egg/plant_protein/meat/seafood as defined. Single dish only; no "X served with Y".

Target cell:
- cube_type={cube_type} (morphology: {target['morphology_name']} — {target['definition']})
- protein_type={protein_type}
- starch_type={starch_type}

Canonical examples for this cube_type: {examples}

Task:
- Propose up to {candidates_per_cell} plausible or novel foods that strictly fit this cell.
- You may invent a new dish name and description, or name a little-known real dish; either way set is_real to false (speculative).
- Same HARD CONSTRAINTS: exact starch_type and protein_type; single canonical entity; no composed plates.
- Keep shortname concise and description to one sentence.
- Return empty list if you cannot think of a plausible fit.

Output JSON only, this shape:
{{
  "candidates": [
    {{
      "shortname": "string",
      "description": "string",
      "is_real": false,
      "confidence": 0.0,
      "rationale_brief": "one short sentence on morphology fit"
    }}
  ]
}}
""".strip()


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        needle = term.lower()
        if not needle:
            continue
        # Use token boundaries to reduce accidental substring hits
        # (e.g., "cream" should not match "creamy").
        pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def passes_axis_lexical_guardrails(
    shortname: str,
    description: str,
    protein_type: str,
    starch_type: str,
    cube_type: str,
) -> bool:
    text = f"{shortname} {description}".lower()

    if contains_any(text, DISALLOWED_COMPOSITION_PHRASES):
        return False

    # Cube Rule canonical morphology: do not place ramen/poutine/etc. in wrong cube_type.
    for term, canonical_cube in CANONICAL_MORPHOLOGY_TERMS.items():
        if canonical_cube != cube_type and contains_any(text, [term]):
            return False

    required_starch_terms = STARCH_REQUIRED_TERMS.get(starch_type, [])
    if required_starch_terms and not contains_any(text, required_starch_terms):
        return False
    forbidden_starch_terms = STARCH_FORBIDDEN_TERMS.get(starch_type, [])
    if forbidden_starch_terms and contains_any(text, forbidden_starch_terms):
        return False

    if protein_type == "none" and contains_any(text, PROTEIN_FORBIDDEN_WHEN_NONE):
        return False
    if protein_type != "none":
        # Reject explicit protein-category conflicts (e.g., seafood terms in a poultry cell).
        # We do not require explicit mention of the target protein token, because many valid
        # canonical dish names/descriptions omit the protein word.
        for category, terms in PROTEIN_CATEGORY_TERMS.items():
            if category == protein_type:
                continue
            if contains_any(text, terms):
                return False
    return True


def all_tensor_cells() -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for cube in CUBE_TYPES:
        for protein in PROTEIN_TYPES:
            for starch in STARCH_TYPES:
                cells.append((cube, protein, starch))
    return cells


def fillable_tensor_cells() -> list[tuple[str, str, str]]:
    """Cells that are allowed to be filled; excludes (salad, *, k) — under the Cube Rule, salad cannot contain starch."""
    return [
        (c, p, s)
        for (c, p, s) in all_tensor_cells()
        if not is_salad_starch_prohibited(c, p, s)
    ]


def occupied_cells(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    occupied: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("review_status") == "rejected":
            continue
        occupied.add((row["cube_type"], row["protein_type"], row["starch_type"]))
    return occupied


def validated_candidates(
    payload: dict[str, Any],
    cube_type: str,
    protein_type: str,
    starch_type: str,
    model: str,
    min_confidence: float,
    speculative: bool = False,
) -> list[dict[str, Any]]:
    if is_salad_starch_prohibited(cube_type, protein_type, starch_type):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Response missing 'candidates' list")
    rows: list[dict[str, Any]] = []
    cube_idx, protein_idx, starch_idx = validate_axes(cube_type, protein_type, starch_type)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        shortname = str(item.get("shortname", "")).strip()
        description = str(item.get("description", "")).strip()
        if not shortname or not description:
            continue
        if not passes_axis_lexical_guardrails(
            shortname=shortname,
            description=description,
            protein_type=protein_type,
            starch_type=starch_type,
            cube_type=cube_type,
        ):
            continue
        confidence_raw = item.get("confidence")
        confidence = None if confidence_raw is None else float(confidence_raw)
        if confidence is None or confidence < min_confidence:
            continue
        is_real = False if speculative else bool(item.get("is_real", False))
        review_status = "accepted" if speculative else "pending"
        record = FoodRecord(
            food_id=normalize_food_id(shortname, cube_type, protein_type, starch_type),
            shortname=shortname,
            description=description,
            is_real=is_real,
            cube_type=cube_type,
            cube_idx=cube_idx,
            protein_type=protein_type,
            protein_idx=protein_idx,
            starch_type=starch_type,
            starch_idx=starch_idx,
            source="llm_generated",
            source_url=None,
            confidence=confidence,
            llm_model=model,
            review_status=review_status,
            rationale_brief=str(item.get("rationale_brief", "")).strip() or None,
        )
        rows.append(record.to_dict())
    return rows


def rows_from_batch_artifact(
    payload: dict[str, Any],
    model: str,
    min_confidence: float,
    speculative: bool,
) -> list[dict[str, Any]]:
    """Reconstruct food rows from a saved batch artifact (for resume)."""
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not item.get("ok") or "raw_response" not in item or "cell" not in item:
            continue
        cell = item["cell"]
        if not isinstance(cell, list) or len(cell) != 3:
            continue
        cube_type, protein_type, starch_type = cell[0], cell[1], cell[2]
        try:
            parsed = extract_tool_input(item["raw_response"])
            rows.extend(
                validated_candidates(
                    parsed,
                    cube_type,
                    protein_type,
                    starch_type,
                    model,
                    min_confidence=min_confidence,
                    speculative=speculative,
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    return rows


def run_batch(
    batch_cells: list[tuple[str, str, str]],
    api_key: str,
    model: str,
    max_tokens: int,
    candidates_per_cell: int,
    max_retries: int,
    min_confidence: float,
    speculative: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated_rows: list[dict[str, Any]] = []
    batch_artifact: list[dict[str, Any]] = []
    for cube_type, protein_type, starch_type in tqdm(
        batch_cells,
        desc="Cells in batch",
        leave=False,
    ):
        if speculative:
            prompt = build_prompt_speculative(cube_type, protein_type, starch_type, candidates_per_cell)
        else:
            prompt = build_prompt(cube_type, protein_type, starch_type, candidates_per_cell)
        last_error = ""
        parsed: dict[str, Any] | None = None
        raw_response: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            try:
                response = call_anthropic(api_key, model, prompt, max_tokens=max_tokens)
                raw_response = response
                parsed = extract_tool_input(response)
                break
            except (
                RuntimeError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = str(exc)
                # Short bounded backoff for network and parse failures.
                time.sleep(1.5 * (attempt + 1))
        if parsed is None:
            batch_artifact.append(
                {
                    "cell": [cube_type, protein_type, starch_type],
                    "ok": False,
                    "error": last_error,
                    "raw_response": raw_response,
                }
            )
            continue
        try:
            rows = validated_candidates(
                parsed,
                cube_type,
                protein_type,
                starch_type,
                model,
                min_confidence=min_confidence,
                speculative=speculative,
            )
            generated_rows.extend(rows)
            batch_artifact.append(
                {
                    "cell": [cube_type, protein_type, starch_type],
                    "ok": True,
                    "candidate_count": len(rows),
                    "raw_response": raw_response,
                }
            )
        except Exception as exc:  # noqa: BLE001
            batch_artifact.append(
                {
                    "cell": [cube_type, protein_type, starch_type],
                    "ok": False,
                    "error": str(exc),
                    "raw_response": raw_response,
                }
            )
    return generated_rows, batch_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill empty tensor cells with Anthropic-generated foods.")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--candidates-per-cell", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Generate novel/plausible food ideas (is_real=false) under same strict rules; no review gate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all empty cells (no max-batches limit). Default is --max-batches 1.",
    )
    args = parser.parse_args()

    api_key = (
        os.getenv("ANTHROPIC_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or _api_key_from_zshrc("ANTHROPIC_KEY", "ANTHROPIC_API_KEY")
    )
    if not api_key:
        log_warn("Missing ANTHROPIC_KEY/ANTHROPIC_API_KEY (env and ~/.zshrc).")
        return 1

    existing = read_jsonl(FOODS_JSONL)
    existing_ids = {row["food_id"] for row in existing}

    # Resume: load any existing speculative batch artifacts into existing so we skip those cells
    if args.speculative and BATCH_DIR.exists():
        n_before = len(existing)
        for path in sorted(BATCH_DIR.glob("*_speculative.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                model_in_file = payload.get("model", args.model)
                batch_rows = rows_from_batch_artifact(
                    payload, model_in_file, args.min_confidence, speculative=True
                )
                for row in batch_rows:
                    if row["food_id"] not in existing_ids:
                        existing.append(row)
                        existing_ids.add(row["food_id"])
            except (json.JSONDecodeError, OSError) as e:
                log_warn(f"Skip batch file {path.name}: {e}")
        if len(existing) > n_before:
            write_jsonl(FOODS_JSONL, existing)
            log_info("Consolidated previous speculative batch results into foods.jsonl")

    occupied = occupied_cells(existing)
    fillable = fillable_tensor_cells()
    empty_cells = [cell for cell in fillable if cell not in occupied]
    batches = chunked(empty_cells, args.batch_size)
    if not args.all:
        batches = batches[: args.max_batches]
    if not batches:
        log_warn("No empty cells to process.")
        return 0

    mode = "speculative (novel ideas)" if args.speculative else "real candidates"
    log_info(
        f"Starting LLM fill [{mode}]: model={args.model}, empty_cells={len(empty_cells)}, "
        f"batches={len(batches)}, batch_size={args.batch_size}, min_confidence={args.min_confidence}"
    )
    ensure_parent(BATCH_DIR / ".keep")
    merged = existing[:]
    added_total = 0
    for batch_num, batch_cells in enumerate(
        tqdm(batches, desc="Batches", unit="batch"),
        start=1,
    ):
        log_info(f"Batch {batch_num}/{len(batches)}: {len(batch_cells)} cells")
        rows, artifact = run_batch(
            batch_cells=batch_cells,
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            candidates_per_cell=args.candidates_per_cell,
            max_retries=args.max_retries,
            min_confidence=args.min_confidence,
            speculative=args.speculative,
        )
        added_this = 0
        for row in rows:
            if row["food_id"] in existing_ids:
                continue
            merged.append(row)
            existing_ids.add(row["food_id"])
            added_this += 1
        added_total += added_this
        write_jsonl(FOODS_JSONL, merged)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = "_speculative" if args.speculative else ""
        batch_path = BATCH_DIR / f"batch_{stamp}_{batch_num:03d}{suffix}.json"
        batch_payload = {
            "model": args.model,
            "batch_number": batch_num,
            "cells": [[c, p, s] for (c, p, s) in batch_cells],
            "items": artifact,
        }
        ensure_parent(batch_path)
        batch_path.write_text(json.dumps(batch_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        log_success(f"Wrote batch artifact: {batch_path} (+{added_this} rows, saved to {FOODS_JSONL.name})")

    kind = "speculative" if args.speculative else "LLM"
    log_success(f"Processed {len(batches)} batch(es), added {added_total} {kind} rows to {FOODS_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
