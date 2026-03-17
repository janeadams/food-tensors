#!/usr/bin/env python3
"""Export and apply review decisions for LLM-generated food rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from termcolor import cprint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensor_food.io_utils import read_jsonl, write_jsonl

FOODS_JSONL = ROOT / "data" / "processed" / "foods.jsonl"
REVIEW_CSV = ROOT / "data" / "processed" / "review_decisions.csv"


def export_pending(rows: list[dict[str, object]]) -> int:
    pending = [
        row
        for row in rows
        if row.get("source") == "llm_generated" and row.get("review_status") == "pending"
    ]
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "food_id",
                "shortname",
                "cube_type",
                "protein_type",
                "starch_type",
                "is_real",
                "decision",
                "notes",
            ],
        )
        writer.writeheader()
        for row in tqdm(pending, desc="Export pending", unit="row"):
            writer.writerow(
                {
                    "food_id": row["food_id"],
                    "shortname": row["shortname"],
                    "cube_type": row["cube_type"],
                    "protein_type": row["protein_type"],
                    "starch_type": row["starch_type"],
                    "is_real": row["is_real"],
                    "decision": "",
                    "notes": "",
                }
            )
    cprint(f"Exported {len(pending)} pending rows to {REVIEW_CSV}", "green")
    return 0


def apply_decisions(rows: list[dict[str, object]]) -> int:
    if not REVIEW_CSV.exists():
        cprint(f"Missing review CSV: {REVIEW_CSV}", "yellow")
        return 1

    decisions: dict[str, str] = {}
    with REVIEW_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        parsed_rows = list(reader)
        for row in tqdm(parsed_rows, desc="Read decisions", unit="row"):
            food_id = (row.get("food_id") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            if not food_id or decision not in {"accepted", "rejected"}:
                continue
            decisions[food_id] = decision

    updated = 0
    for row in tqdm(rows, desc="Apply decisions", unit="row"):
        decision = decisions.get(str(row.get("food_id", "")))
        if not decision:
            continue
        if row.get("source") != "llm_generated":
            continue
        row["review_status"] = decision
        updated += 1

    write_jsonl(FOODS_JSONL, rows)
    cprint(f"Applied {updated} decision(s) to {FOODS_JSONL}", "green")
    return 0


def triage_undecided(rows: list[dict[str, object]]) -> int:
    """Second-pass triage: set decision/notes for rows still undecided using confidence and description."""
    if not REVIEW_CSV.exists():
        cprint(f"Missing review CSV: {REVIEW_CSV}", "yellow")
        return 1

    by_id = {str(r.get("food_id", "")): r for r in rows if r.get("source") == "llm_generated"}
    with REVIEW_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        table = list(reader)

    # Second-pass rules: reject conf <= 0.79 or "served with" in description; accept conf >= 0.80 otherwise
    PLATE_PHRASE = "served with"
    updated = 0
    for row in table:
        decision = (row.get("decision") or "").strip()
        if decision:
            continue
        food_id = (row.get("food_id") or "").strip()
        rec = by_id.get(food_id)
        if not rec:
            continue
        conf = rec.get("confidence")
        desc = (rec.get("description") or "").lower()
        base_note = (row.get("notes") or "").strip()

        if conf is None:
            row["notes"] = f"{base_note} Second-pass: no confidence; left undecided.".strip()
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            row["notes"] = f"{base_note} Second-pass: invalid confidence; left undecided.".strip()
            continue

        if PLATE_PHRASE in desc:
            row["decision"] = "rejected"
            row["notes"] = f"{base_note} Second-pass: plate-combo phrasing in description.".strip()
            updated += 1
            continue
        if c <= 0.79:
            row["decision"] = "rejected"
            row["notes"] = f"{base_note} Second-pass: low confidence ({c}) for strict mode.".strip()
            updated += 1
            continue
        if c >= 0.80:
            row["decision"] = "accepted"
            row["notes"] = f"{base_note} Second-pass: accepted (confidence {c}).".strip()
            updated += 1
            continue
        row["notes"] = f"{base_note} Second-pass: confidence {c} in (0.79, 0.80); left undecided.".strip()

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    cprint(f"Second-pass triage: set {updated} decision(s) in {REVIEW_CSV}", "green")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and apply LLM review decisions.")
    parser.add_argument(
        "--mode",
        choices=["export", "apply", "triage"],
        required=True,
        help="export pending to CSV, apply decisions from CSV, or triage undecided rows",
    )
    args = parser.parse_args()

    rows = read_jsonl(FOODS_JSONL)
    if args.mode == "export":
        return export_pending(rows)
    if args.mode == "triage":
        return triage_undecided(rows)
    return apply_decisions(rows)


if __name__ == "__main__":
    raise SystemExit(main())
