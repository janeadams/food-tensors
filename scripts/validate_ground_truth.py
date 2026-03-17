#!/usr/bin/env python3
"""Validate canonical Cube Rule examples against pipeline guardrails."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from termcolor import cprint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fill_tensor_llm as pipeline  # noqa: E402
from tensor_food.schema import is_structurally_invalid_cell

SEED_CSV = ROOT / "data" / "raw" / "cube_rule_examples.csv"
FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"


def main() -> int:
    if not SEED_CSV.exists():
        cprint(f"Missing seed CSV: {SEED_CSV}", "yellow")
        return 1

    canonical_ids: set[str] = set()
    if FOODS_JSONL.exists():
        for line in FOODS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "cube_rule_prefill":
                canonical_ids.add(str(row.get("food_id", "")))

    failures: list[tuple[str, str]] = []
    total = 0
    in_processed = 0

    with SEED_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(list(reader), desc="Ground-truth checks", unit="row"):
            shortname = row["shortname"].strip()
            description = row["description"].strip()
            cube_type = row["cube_type"].strip()
            protein_type = row["protein_type"].strip()
            starch_type = row["starch_type"].strip()

            if is_structurally_invalid_cell(cube_type, protein_type, starch_type):
                continue  # skip: cube/starch pair is definitionally invalid under the ontology
            total += 1

            # Check axis vocabulary validity.
            try:
                pipeline.validate_axes(cube_type, protein_type, starch_type)
            except Exception as exc:  # noqa: BLE001
                failures.append((shortname, f"invalid axis mapping: {exc}"))
                continue

            # Check the same lexical guardrails used by generated candidates.
            if not pipeline.passes_axis_lexical_guardrails(
                shortname=shortname,
                description=description,
                protein_type=protein_type,
                starch_type=starch_type,
                cube_type=cube_type,
            ):
                failures.append((shortname, "failed lexical guardrails"))

            # Check that canonical seed exists in canonical processed set.
            food_id = pipeline.normalize_food_id(shortname, cube_type, protein_type, starch_type)
            if food_id in canonical_ids:
                in_processed += 1
            else:
                failures.append((shortname, f"missing canonical id in processed foods: {food_id}"))

    cprint(f"Canonical examples checked: {total}", "cyan")
    cprint(f"Canonical examples present in processed set: {in_processed}/{total}", "cyan")
    cprint(f"Guardrail failures: {len(failures)}", "green" if not failures else "yellow")
    for shortname, reason in failures:
        cprint(f"- {shortname}: {reason}", "yellow")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
