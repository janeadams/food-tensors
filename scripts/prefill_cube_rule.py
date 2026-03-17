#!/usr/bin/env python3
"""Normalize Cube Rule seed CSV into canonical foods JSONL."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from termcolor import cprint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensor_food.io_utils import write_jsonl
from tensor_food.schema import FoodRecord, normalize_food_id, validate_axes

SEED_CSV = ROOT / "data" / "raw" / "cube_rule_examples.csv"
FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"


def parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


def main() -> int:
    if not SEED_CSV.exists():
        cprint(f"Missing seed file: {SEED_CSV}", "yellow")
        return 1

    rows: list[dict[str, object]] = []
    with SEED_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
        for source_row in tqdm(reader, desc="Prefill rows", unit="row"):
            cube_type = source_row["cube_type"].strip()
            protein_type = source_row["protein_type"].strip()
            starch_type = source_row["starch_type"].strip()
            cube_idx, protein_idx, starch_idx = validate_axes(cube_type, protein_type, starch_type)
            shortname = source_row["shortname"].strip()
            record = FoodRecord(
                food_id=normalize_food_id(shortname, cube_type, protein_type, starch_type),
                shortname=shortname,
                description=source_row["description"].strip(),
                is_real=parse_bool(source_row["is_real"]),
                cube_type=cube_type,
                cube_idx=cube_idx,
                protein_type=protein_type,
                protein_idx=protein_idx,
                starch_type=starch_type,
                starch_idx=starch_idx,
                source="cube_rule_prefill",
                source_url=source_row.get("source_url", "").strip() or None,
                confidence=None,
                llm_model=None,
                review_status="accepted",
                rationale_brief="Canonical Cube Rule prefill seed.",
            )
            rows.append(record.to_dict())

    write_jsonl(FOODS_JSONL, rows)
    cprint(f"Wrote {len(rows)} seed records to {FOODS_JSONL}", "green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
