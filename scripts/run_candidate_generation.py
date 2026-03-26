"""
run_candidate_generation.py

Role
----
CLI entry point for prompt-free candidate generation.

Outputs
-------
- candidate masks
- overlay image
- candidate manifest JSON
- optional preview table for later review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.segmentation.sam_backend import SAMBackend
from ni_analysis.segmentation.candidate_generator import CandidateGenerator
from ni_analysis.utils.io_utils import ensure_dir, save_mask_png, save_overlay_png


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = ensure_dir(args.output_dir)
    masks_dir = ensure_dir(output_dir / "candidate_masks")

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
    )

    batch = generator.generate(
        image=image,
        source_image_id=args.image_id,
        apply_preprocessing=True,
    )

    saved_mask_paths: list[str] = []
    for cand in batch.candidates:
        mask_path = masks_dir / f"{cand.candidate_id}.png"
        save_mask_png(cand.mask, mask_path)
        saved_mask_paths.append(str(mask_path))

    overlay_path = output_dir / "candidate_overlay.png"
    save_overlay_png(
        image=batch.image_rgb,
        masks=[c.mask for c in batch.candidates],
        save_path=overlay_path,
        alpha=0.45,
        draw_boundaries=True,
    )

    manifest = {
        "source_image_id": batch.source_image_id,
        "source_image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "candidate_count": len(batch.candidates),
        "run_metadata": batch.run_metadata,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "score": c.score,
                "area_px": c.area_px,
                "bbox_xyxy": c.bbox_xyxy,
                "mask_path": saved_mask_paths[idx],
                "source_image_id": c.source_image_id,
                "metadata": c.metadata,
            }
            for idx, c in enumerate(batch.candidates)
        ],
    }

    manifest_path = output_dir / "candidate_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] Candidate generation completed.")
    print(f"Candidates : {len(batch.candidates)}")
    print(f"Overlay    : {overlay_path}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()