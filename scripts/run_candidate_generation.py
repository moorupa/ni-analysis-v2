"""
Path
----
ni-analysis-v2/scripts/run_candidate_generation.py

Role
----
CLI entry point for prompt-free candidate generation.

Outputs
-------
- candidate masks
- overlay image
- candidate manifest JSON
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.segmentation.sam_backend import SAMBackend
from ni_analysis.segmentation.candidate_generator import CandidateGenerator
from ni_analysis.utils.io_utils import (
    build_candidate_mask_path,
    save_candidate_manifest,
    save_mask,
    save_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt-free candidate generation for SEM images."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to source image.")
    parser.add_argument("--image-id", type=str, required=True, help="Logical image/sample ID.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu.")
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--sampling-grid-size", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.70)
    parser.add_argument("--stability-score-thresh", type=float, default=0.70)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    backend = SAMBackend(
        checkpoint_path=args.checkpoint or None,
        device=args.device,
    )

    generator = CandidateGenerator(
        backend=backend,
        min_area=args.min_area,
        max_area=args.max_area,
        kernel_size=args.kernel_size,
        remove_border_touching=True,
        sampling_grid_size=args.sampling_grid_size,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,        
    )

    batch = generator.generate(
        image=image,
        source_image_id=args.image_id,
        apply_preprocessing=True,
    )

    candidate_rows: list[dict] = []
    saved_masks: list = []

    for cand in batch.candidates:
        mask_path = build_candidate_mask_path(output_dir, cand.candidate_id)
        save_mask(cand.mask, mask_path)
        saved_masks.append(cand.mask)

        candidate_rows.append(
            {
                "candidate_id": cand.candidate_id,
                "score": cand.score,
                "area_px": cand.area_px,
                "bbox_xyxy": cand.bbox_xyxy,
                "mask_path": str(mask_path),
                "source_image_id": cand.source_image_id,
                "metadata": cand.metadata,
            }
        )

    overlay_path = output_dir / "candidate_overlay.png"
    save_overlay(
        image=batch.image_rgb,
        masks=saved_masks,
        path=overlay_path,
        alpha=0.45,
        draw_boundaries=True,
    )

    manifest_path = output_dir / "candidate_manifest.json"
    save_candidate_manifest(
        source_image_id=batch.source_image_id,
        source_image_path=str(image_path),
        overlay_path=str(overlay_path),
        run_metadata=batch.run_metadata,
        candidates=candidate_rows,
        path=manifest_path,
    )

    print("[DONE] Candidate generation completed.")
    print(f"Candidates : {len(batch.candidates)}")
    print(f"Overlay    : {overlay_path}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()