"""
candidate_generator.py

Role
----
High-level orchestration for prompt-free candidate generation.

Responsibilities
----------------
1) call SAM backend
2) postprocess masks
3) assign candidate IDs
4) attach lightweight per-candidate metadata
5) prepare outputs for review stage

This file should be the main segmentation-side entry point for v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ni_analysis.segmentation.sam_backend import SAMBackend
from ni_analysis.segmentation.mask_postprocess import postprocess_masks


@dataclass
class CandidateRecord:
    candidate_id: str
    mask: np.ndarray
    score: float | None
    area_px: int
    bbox_xyxy: tuple[int, int, int, int] | None
    source_image_id: str
    metadata: dict[str, Any]


@dataclass
class CandidateBatch:
    source_image_id: str
    image_rgb: np.ndarray
    candidates: list[CandidateRecord]
    run_metadata: dict[str, Any]


def compute_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class CandidateGenerator:
    """
    Prompt-free candidate generation pipeline for a single image.
    """

    def __init__(
        self,
        backend: SAMBackend,
        min_area: int = 50,
        max_area: int | None = None,
        kernel_size: int = 3,
        remove_border_touching: bool = True,
    ) -> None:
        self.backend = backend
        self.min_area = min_area
        self.max_area = max_area
        self.kernel_size = kernel_size
        self.remove_border_touching = remove_border_touching

    def generate(
        self,
        image: Image.Image,
        source_image_id: str,
        apply_preprocessing: bool = True,
    ) -> CandidateBatch:
        """
        Generate review-ready candidates from one image.
        """
        result = self.backend.generate_candidates(
            image=image,
            apply_preprocessing=apply_preprocessing,
        )

        cleaned_masks = postprocess_masks(
            masks=result.masks,
            min_area=self.min_area,
            max_area=self.max_area,
            kernel_size=self.kernel_size,
            do_open=True,
            do_close=True,
            fill_holes=True,
            remove_border_touching=self.remove_border_touching,
        )

        candidates: list[CandidateRecord] = []
        for idx, mask in enumerate(cleaned_masks, start=1):
            bbox = compute_bbox_xyxy(mask)
            area_px = int(np.asarray(mask).sum())
            raw_score = result.scores[idx - 1] if idx - 1 < len(result.scores) else None

            candidates.append(
                CandidateRecord(
                    candidate_id=f"{source_image_id}_cand_{idx:04d}",
                    mask=np.asarray(mask).astype(np.uint8),
                    score=None if raw_score is None else float(raw_score),
                    area_px=area_px,
                    bbox_xyxy=bbox,
                    source_image_id=source_image_id,
                    metadata={},
                )
            )

        run_metadata = {
            **result.metadata,
            "source_image_id": source_image_id,
            "raw_candidate_count": len(result.masks),
            "postprocessed_candidate_count": len(candidates),
        }

        return CandidateBatch(
            source_image_id=source_image_id,
            image_rgb=result.image_rgb,
            candidates=candidates,
            run_metadata=run_metadata,
        )