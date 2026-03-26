"""
legacy_run_text_prompt_baseline.py

Role
----
Legacy baseline runner for the old prompt-dependent segmentation path.

This script is intentionally NOT the main workflow of ni-analysis-v2.
It exists only for comparison against the new prompt-free pipeline.
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
from ni_analysis.segmentation.mask_postprocess import postprocess_masks
from ni_analysis.utils.io_utils import ensure_dir, save_mask_png, save_overlay_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy prompt-dependent SAM baseline runner."
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for legacy baseline")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = ensure_dir(args.output_dir)
    masks_dir = ensure_dir(output_dir / "masks")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    backend = SAMBackend(
        checkpoint_path=args.checkpoint or None,
        device=args.device,
    )

    result = backend.segment_from_text(
        image=image,
        prompt=args.prompt,
        apply_preprocessing=True,
    )

    cleaned_masks = postprocess_masks(
        masks=result.masks,
        min_area=args.min_area,
        max_area=args.max_area,
        kernel_size=args.kernel_size,
        do_open=True,
        do_close=True,
        fill_holes=True,
        remove_border_touching=True,
    )

    saved_mask_paths: list[str] = []
    for idx, mask in enumerate(cleaned_masks, start=1):
        mask_path = masks_dir / f"mask_{idx:04d}.png"
        save_mask_png(mask, mask_path)
        saved_mask_paths.append(str(mask_path))

    overlay_path = output_dir / "overlay.png"
    save_overlay_png(
        image=result.image_rgb,
        masks=cleaned_masks,
        save_path=overlay_path,
        alpha=0.45,
        draw_boundaries=True,
    )

    metadata = {
        "mode": "legacy_text_prompt_baseline",
        "image_path": str(image_path),
        "prompt": args.prompt,
        "raw_mask_count": len(result.masks),
        "postprocessed_mask_count": len(cleaned_masks),
        "mask_paths": saved_mask_paths,
        "overlay_path": str(overlay_path),
        "backend_metadata": result.metadata,
    }

    metadata_path = output_dir / "legacy_baseline_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("[DONE] Legacy baseline completed.")
    print(f"Raw masks         : {len(result.masks)}")
    print(f"Postprocessed     : {len(cleaned_masks)}")
    print(f"Overlay           : {overlay_path}")
    print(f"Metadata          : {metadata_path}")


if __name__ == "__main__":
    main()