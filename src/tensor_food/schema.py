"""Canonical schema and axis definitions for tensor food records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CUBE_TYPES = [
    "salad",
    "toast",
    "sandwich",
    "taco",
    "sushi",
    "quiche",
    "calzone",
    "cake",
    "nachos",
]

PROTEIN_TYPES = [
    "none",
    "red_meat",
    "poultry",
    "seafood",
    "egg",
    "plant_protein",
    "dairy",
]

STARCH_TYPES = [
    "wheat",
    "rice",
    "corn",
    "potato",
    "pasta_noodle",
    "pastry_dough",
    "other_grain",
    "none",
]

SOURCE_TYPES = {"cube_rule_prefill", "llm_generated", "manual"}
REVIEW_STATUSES = {"pending", "accepted", "rejected"}

CUBE_INDEX = {name: idx for idx, name in enumerate(CUBE_TYPES)}
PROTEIN_INDEX = {name: idx for idx, name in enumerate(PROTEIN_TYPES)}
STARCH_INDEX = {name: idx for idx, name in enumerate(STARCH_TYPES)}


def is_salad_starch_prohibited(cube_type: str, protein_type: str, starch_type: str) -> bool:
    """True if a salad cell is paired with a non-none starch type."""
    return cube_type == "salad" and starch_type != "none"


def is_structurally_invalid_cell(cube_type: str, protein_type: str, starch_type: str) -> bool:
    """True if cube morphology and starch axis disagree about structural starch."""
    return (cube_type == "salad" and starch_type != "none") or (
        cube_type != "salad" and starch_type == "none"
    )


@dataclass(frozen=True)
class FoodRecord:
    food_id: str
    shortname: str
    description: str
    is_real: bool
    cube_type: str
    cube_idx: int
    protein_type: str
    protein_idx: int
    starch_type: str
    starch_idx: int
    source: str
    source_url: str | None
    confidence: float | None
    llm_model: str | None
    review_status: str
    rationale_brief: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "food_id": self.food_id,
            "shortname": self.shortname,
            "description": self.description,
            "is_real": self.is_real,
            "cube_type": self.cube_type,
            "cube_idx": self.cube_idx,
            "protein_type": self.protein_type,
            "protein_idx": self.protein_idx,
            "starch_type": self.starch_type,
            "starch_idx": self.starch_idx,
            "source": self.source,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "llm_model": self.llm_model,
            "review_status": self.review_status,
            "rationale_brief": self.rationale_brief,
        }


def normalize_food_id(shortname: str, cube_type: str, protein_type: str, starch_type: str) -> str:
    normalized = shortname.strip().lower().replace("&", "and").replace("/", " ")
    pieces = ["".join(ch for ch in token if ch.isalnum()) for token in normalized.split()]
    head = "_".join(piece for piece in pieces if piece)
    return f"{head}_{cube_type}_{protein_type}_{starch_type}"


def validate_axes(cube_type: str, protein_type: str, starch_type: str) -> tuple[int, int, int]:
    if cube_type not in CUBE_INDEX:
        raise ValueError(f"Unknown cube_type: {cube_type}")
    if protein_type not in PROTEIN_INDEX:
        raise ValueError(f"Unknown protein_type: {protein_type}")
    if starch_type not in STARCH_INDEX:
        raise ValueError(f"Unknown starch_type: {starch_type}")
    return CUBE_INDEX[cube_type], PROTEIN_INDEX[protein_type], STARCH_INDEX[starch_type]


def validate_record_payload(payload: dict[str, Any]) -> None:
    required = {
        "food_id",
        "shortname",
        "description",
        "is_real",
        "cube_type",
        "cube_idx",
        "protein_type",
        "protein_idx",
        "starch_type",
        "starch_idx",
        "source",
        "review_status",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    validate_axes(payload["cube_type"], payload["protein_type"], payload["starch_type"])
    if payload["source"] not in SOURCE_TYPES:
        raise ValueError(f"Unknown source: {payload['source']}")
    if payload["review_status"] not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review_status: {payload['review_status']}")
