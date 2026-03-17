# A Tensor-Based Expansion of the Cube Rule for Generative Nutrient Morphology

## Quick Start

1. Sync the project environment:
   ```bash
   uv sync
   ```
2. Seed canonical foods from Cube Rule examples:
   ```bash
   uv run python scripts/prefill_cube_rule.py
   ```
3. Generate paper figures:
   ```bash
   uv run python scripts/plot_lettuce_croutons_gradient.py
   uv run python scripts/plot_tensor.py --include-3d --export-3d-png
   uv run python scripts/plot_confidence_histogram.py
   ```
4. Generate optional experiment outputs used by the paper when available:
   ```bash
   ANTHROPIC_KEY=... uv run python scripts/salad_cube_rule_experiment.py
   ```
5. Build the paper PDF:
   ```bash
   uv run python scripts/build_paper.py
   ```
6. Open outputs:
   - `paper/main.pdf`
   - `results/figures/canonical_plus_llm_identified_plus_llm_speculated/tensor_3d.html`

## Repository Layout

- `paper/`: LaTeX source.
- `paper/sections/`: section files included by `paper/main.tex`.
- `src/tensor_food/`: shared schema and JSONL helpers.
- `scripts/`: pipeline and utility scripts.
- `data/raw/`: immutable source datasets.
- `data/processed/`: canonical foods and review tables.
- `results/`: generated outputs and experiment artifacts.

## Workflow

1. Prefill canonical examples with `uv run python scripts/prefill_cube_rule.py`.
2. Fill empty cells with LLM suggestions:
   - `ANTHROPIC_KEY=... uv run python scripts/fill_tensor_llm.py --max-batches 2`
3. Review LLM outputs:
   - export: `uv run python scripts/review_llm_outputs.py --mode export`
   - edit `data/processed/review_decisions.csv` with `accepted` or `rejected`
   - apply: `uv run python scripts/review_llm_outputs.py --mode apply`
4. Render tensor visualizations with `uv run python scripts/plot_tensor.py --include-3d --export-3d-png`.
5. Render supporting figures with `uv run python scripts/plot_lettuce_croutons_gradient.py` and `uv run python scripts/plot_confidence_histogram.py`.
6. Generate the salad mini-experiment table with `ANTHROPIC_KEY=... uv run python scripts/salad_cube_rule_experiment.py` when you want that section populated.
7. Edit section source in `paper/sections/` and compile with `uv run python scripts/build_paper.py`.

## Project Scope

This repository implements a tensor-based extension of the Cube Rule of Food.
Code, data generation, and analysis outputs are organized so they can be developed in parallel with the writeup.
