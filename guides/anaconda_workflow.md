# Anaconda Workflow

This repository is currently run from Anaconda Prompt with the `euang` environment.

## 1. Activate environment

```bat
conda activate euang
cd C:\Users\admin\Documents\GitHub\ni-analysis-v2
```

## 2. Run SAM candidate generation

Use `--model-type vit_b` with the `sam_vit_b_01ec64.pth` checkpoint.

```bat
python scripts\run_candidate_generation.py ^
  --image "C:\path\to\image.jpg" ^
  --image-id "sample_01" ^
  --output-dir "results\candidate_runs\sample_01" ^
  --checkpoint "C:\Users\admin\Desktop\checkpoint\sam_hamington\checkpoints\sam_vit_b_01ec64.pth" ^
  --model-type vit_b ^
  --device cuda ^
  --sampling-grid-size 8
```

The manifest should report:

```text
backend_mode: sam_automatic_mask_generator
```

## 3. Fallback candidate generation

Use this only when checking the pipeline without SAM.

```bat
python scripts\run_candidate_generation.py ^
  --image "C:\path\to\image.jpg" ^
  --image-id "sample_01_fallback" ^
  --output-dir "results\candidate_runs\sample_01_fallback" ^
  --model-type fallback
```

The manifest should report:

```text
backend_mode: fallback_connected_components
```

## 4. Review, export, and extract features

```bat
python scripts\review_session.py init ^
  --manifest results\candidate_runs\sample_01\candidate_manifest.json ^
  --output-dir results\review_sessions\sample_01 ^
  --session-id sample_01_r1 ^
  --reviewer-id bm

python scripts\review_session.py update ^
  --session results\review_sessions\sample_01\sample_01_r1.review.json ^
  --candidate-id sample_01_cand_0001 ^
  --review-label accept_single ^
  --morphology-label spiky ^
  --confidence 5

python scripts\export_reviewed_dataset.py ^
  --session results\review_sessions\sample_01\sample_01_r1.review.json ^
  --output-dir results\reviewed\sample_01

python scripts\extract_features_from_reviewed.py ^
  --reviewed-csv results\reviewed\sample_01\reviewed_dataset.csv ^
  --output-dir results\features\sample_01
```

