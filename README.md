# ni-analysis-v2

`ni-analysis-v2` is a **materials-characterization research pipeline** for SEM images of electroless Ni particles (especially rough/spiky/urchin-like morphologies).

The goal is to build a reproducible workflow for:

1. prompt-free candidate generation,
2. human review,
3. reviewed dataset export,
4. morphology/roughness/spikiness feature extraction,
5. incremental extension toward objectness priors and VLM-assisted ranking.

This repository intentionally prioritizes **inspectable research plumbing** over end-to-end model complexity.

## Current pipeline (v0.1)

1. **Candidate generation** (`ni-candidates`)
   - Uses `SAMBackend` interface.
   - Today, when no SAM model is wired, it falls back to a lightweight connected-component proposal path so the pipeline remains runnable.
2. **Review session management** (`ni-review`)
   - Initialize review JSON from a candidate manifest.
   - List IDs, view summary, update decisions.
3. **Reviewed dataset export** (`ni-export-reviewed`)
   - Exports accepted masks (or all reviewed masks) and metadata CSV/manifest.
4. **Feature extraction** (`ni-extract-features`)
   - Computes morphology + roughness + spikiness descriptors from reviewed masks.
5. **Objectness prior (optional placeholder)**
   - `ni-export-objectness` builds candidate-level metadata rows from manifest+review.
   - `ni-train-objectness` trains a small metadata MLP when PyTorch is available.

## Repository structure

- `src/ni_analysis/segmentation/` — candidate generation orchestration + backend wrappers.
- `src/ni_analysis/review/` — review schema, persistence, session update logic.
- `src/ni_analysis/export/` — reviewed dataset export.
- `src/ni_analysis/features/` — morphology/roughness/spikiness feature extraction.
- `src/ni_analysis/prior/` — metadata objectness prior model.
- `src/ni_analysis/utils/` — shared JSON/image/mask/config I/O utilities.
- `scripts/` — script entrypoints (also exposed as package console commands).
- `configs/` — example YAML configs.

## Installation

### 1) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install package

```bash
pip install -e .
```

### 3) (Optional) install test tooling

```bash
pip install -e .[dev]
```

## CLI examples

### a) Candidate generation

```bash
ni-candidates \
  --image data/sem/example.png \
  --image-id sem_example_001 \
  --output-dir runs/sem_example_001/candidates
```

Outputs:
- `candidate_masks/*.png`
- `candidate_overlay.png`
- `candidate_manifest.json`

### b) Initialize review session

```bash
ni-review init \
  --manifest runs/sem_example_001/candidates/candidate_manifest.json \
  --output-dir runs/sem_example_001/review \
  --session-id sem_example_001_r1 \
  --reviewer-id bm
```

### c) List / update review decisions

List candidate IDs:

```bash
ni-review list --session runs/sem_example_001/review/sem_example_001_r1.review.json
```

Show summary:

```bash
ni-review summary --session runs/sem_example_001/review/sem_example_001_r1.review.json
```

Update one decision:

```bash
ni-review update \
  --session runs/sem_example_001/review/sem_example_001_r1.review.json \
  --candidate-id sem_example_001_cand_0001 \
  --review-label accept_single \
  --morphology-label spiky \
  --confidence 5 \
  --comment "clear urchin-like spikes"
```

### d) Export reviewed dataset

Accepted-only export (default):

```bash
ni-export-reviewed \
  --session runs/sem_example_001/review/sem_example_001_r1.review.json \
  --output-dir runs/sem_example_001/reviewed
```

Include rejected too:

```bash
ni-export-reviewed \
  --session runs/sem_example_001/review/sem_example_001_r1.review.json \
  --output-dir runs/sem_example_001/reviewed_all \
  --include-rejected
```

### e) Extract morphology features

```bash
ni-extract-features \
  --reviewed-csv runs/sem_example_001/reviewed/reviewed_dataset.csv \
  --output-dir runs/sem_example_001/features
```

### f) Objectness prior (if used)

Export metadata training table:

```bash
ni-export-objectness \
  --manifest runs/sem_example_001/candidates/candidate_manifest.json \
  --session runs/sem_example_001/review/sem_example_001_r1.review.json \
  --output runs/sem_example_001/objectness/objectness_dataset.json
```

Train metadata prior (requires PyTorch):

```bash
ni-train-objectness \
  --dataset runs/sem_example_001/objectness/objectness_dataset.json \
  --output-dir runs/sem_example_001/objectness/model \
  --epochs 30
```

## File formats

### Candidate manifest (`candidate_manifest.json`)

- Top-level keys:
  - `source_image_id` (str)
  - `source_image_path` (str)
  - `overlay_path` (str)
  - `candidate_count` (int)
  - `run_metadata` (dict)
  - `candidates` (list)
- Each candidate item includes:
  - `candidate_id`, `score`, `area_px`, `bbox_xyxy`, `mask_path`, `metadata`.

### Review session (`*.review.json`)

- Top-level keys:
  - `session_id`, `source_image_id`, `source_image_path`
  - `decisions` (list of decision records)
  - `session_metadata`
- Decision fields include:
  - `candidate_id`, `review_label`, `morphology_label`, `confidence`, `comment`
  - `source_mask_path`, `edited_mask_path`, `bbox_xyxy`, `extra_metadata`.

### Reviewed dataset export

- `reviewed_dataset.csv` with one row per exported candidate.
- `reviewed_dataset_manifest.json` summary metadata.
- `reviewed_masks/*.png` binary mask images.

## Developer notes / research TODO

1. **Objectness prior from reviewed candidates**
   - Improve label mapping (`accept_single` vs reject variants) and calibration.
2. **Spiky-particle morphology metrics**
   - Add radial signature descriptors, tip density, branch-length distribution, and robustness to partial masks.
3. **VLM-assisted ranking/prompt proposal**
   - Use reviewed sets for candidate ordering and active-review suggestion.
4. **Downstream dataset export**
   - Add task-specific exporters (e.g., segmentation finetune JSON/COCO-like variants) while preserving scientific metadata lineage.

## Scope and philosophy

This is **not** a generic segmentation demo repository. It is a staged, auditable pipeline for Ni-particle morphology research in SEM workflows.
