"""
Path
----
ni-analysis-v2/src/ni_analysis/segmentation/candidate_generator.py

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ni_analysis.segmentation.sam_backend import SAMBackend
from ni_analysis.segmentation.mask_postprocess import (
    PostprocessConfig,
    bbox_xyxy,
    mask_area,
    postprocess_candidate_masks,
)


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


class CandidateGenerator:
    def __init__(
        self,
        backend: SAMBackend,
        min_area: int = 50,
        max_area: int | None = None,
        kernel_size: int = 3,
        remove_border_touching: bool = True,
        sampling_grid_size: int = 32,
        pred_iou_thresh: float = 0.70,
        stability_score_thresh: float = 0.70,
        min_mask_region_area: int = 0,
    ) -> None:
        self.backend = backend
        self.min_area = min_area
        self.max_area = max_area
        self.kernel_size = kernel_size
        self.remove_border_touching = remove_border_touching
        self.sampling_grid_size = sampling_grid_size
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

    def generate(
        self,
        image: Image.Image,
        source_image_id: str,
        apply_preprocessing: bool = True,
    ) -> CandidateBatch:
        result = self.backend.generate_candidates(
            image=image,
            apply_preprocessing=apply_preprocessing,
            sampling_grid_size=self.sampling_grid_size,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

        config = PostprocessConfig(
            min_area=self.min_area,
            max_area=self.max_area,
            kernel_size=self.kernel_size,
            remove_border_touching=self.remove_border_touching,
        )

        cleaned_masks = postprocess_candidate_masks(result.masks, config=config)

        candidates: list[CandidateRecord] = []
        for idx, mask in enumerate(cleaned_masks, start=1):
            bbox = bbox_xyxy(mask)
            area_px = mask_area(mask)
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