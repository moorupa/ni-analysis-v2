"""
Path
----
ni-analysis-v2/scripts/legacy_run_text_prompt_baseline.py

Role
----
Legacy prompt-dependent segmentation baseline runner.

This is NOT the main workflow of ni-analysis-v2.
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

from ni_analysis.segmentation.legacy_prompt_segmenter import LegacyPromptSegmenter
from ni_analysis.segmentation.mask_postprocess import (
    PostprocessConfig,
    postprocess_candidate_masks,
)
from ni_analysis.utils.io_utils import (
    build_candidate_mask_path,
    save_candidate_manifest,
    save_mask,
    save_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy prompt-dependent segmentation baseline."
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-id", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    segmenter = LegacyPromptSegmenter(
        checkpoint_path=args.checkpoint or None,
        device=args.device,
    )

    result = segmenter.segment(
        image=image,
        prompt=args.prompt,
        apply_preprocessing=True,
    )

    config = PostprocessConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        kernel_size=args.kernel_size,
        remove_border_touching=True,
    )

    cleaned_masks = postprocess_candidate_masks(result.masks, config=config)

    candidate_rows: list[dict] = []
    saved_masks: list = []

    for idx, mask in enumerate(cleaned_masks, start=1):
        candidate_id = f"{args.image_id}_legacy_{idx:04d}"
        mask_path = build_candidate_mask_path(output_dir, candidate_id)
        save_mask(mask, mask_path)
        saved_masks.append(mask)

        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "score": None,
                "area_px": int(mask.sum()),
                "bbox_xyxy": None,
                "mask_path": str(mask_path),
                "source_image_id": args.image_id,
                "metadata": {"mode": "legacy_text_prompt_baseline", "prompt": args.prompt},
            }
        )

    overlay_path = output_dir / "legacy_overlay.png"
    save_overlay(
        image=result.image_rgb,
        masks=saved_masks,
        path=overlay_path,
        alpha=0.45,
        draw_boundaries=True,
    )

    manifest_path = output_dir / "legacy_candidate_manifest.json"
    save_candidate_manifest(
        source_image_id=args.image_id,
        source_image_path=str(image_path),
        overlay_path=str(overlay_path),
        run_metadata=result.metadata,
        candidates=candidate_rows,
        path=manifest_path,
    )

    print("[DONE] Legacy baseline completed.")
    print(f"Candidates : {len(cleaned_masks)}")
    print(f"Overlay    : {overlay_path}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()