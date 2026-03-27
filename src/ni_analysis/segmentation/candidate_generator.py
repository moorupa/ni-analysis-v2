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
5) optionally score candidates with learned objectness prior
6) prepare outputs for review stage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import cv2
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


def _compute_mask_metadata(mask: np.ndarray, image_shape_hw: tuple[int, int]) -> dict[str, Any]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = image_shape_hw

    area_px = int(mask_u8.sum())
    bbox = bbox_xyxy(mask_u8)
    if bbox is None:
        return {
            "area_px": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "aspect_ratio": 0.0,
            "extent": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "is_border_touching": False,
        }

    x1, y1, x2, y2 = bbox
    bw = max(0, int(x2) - int(x1))
    bh = max(0, int(y2) - int(y1))
    bbox_area = max(1, bw * bh)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    contour_area = float(area_px)
    solidity = 0.0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(cnt, True))
        contour_area = float(cv2.contourArea(cnt))

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 1e-8 else 0.0

    circularity = 0.0
    if perimeter > 1e-8:
        circularity = float(4.0 * math.pi * contour_area / (perimeter * perimeter))

    is_border_touching = bool(
        mask_u8[0, :].any()
        or mask_u8[-1, :].any()
        or mask_u8[:, 0].any()
        or mask_u8[:, -1].any()
    )

    return {
        "area_px": area_px,
        "bbox_width": bw,
        "bbox_height": bh,
        "aspect_ratio": float(bw / bh) if bh > 0 else 0.0,
        "extent": float(area_px / bbox_area),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "is_border_touching": is_border_touching,
    }


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
        prior_scorer: Any | None = None,
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
        self.prior_scorer = prior_scorer

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

            metadata = _compute_mask_metadata(mask, image_shape_hw=result.image_rgb.shape[:2])
            metadata["raw_sam_score"] = None if raw_score is None else float(raw_score)

            candidates.append(
                CandidateRecord(
                    candidate_id=f"{source_image_id}_cand_{idx:04d}",
                    mask=np.asarray(mask).astype(np.uint8),
                    score=None if raw_score is None else float(raw_score),
                    area_px=area_px,
                    bbox_xyxy=bbox,
                    source_image_id=source_image_id,
                    metadata=metadata,
                )
            )

        if self.prior_scorer is not None:
            candidates = self.prior_scorer.score_batch(candidates)
            candidates = sorted(
                candidates,
                key=lambda c: c.metadata.get("particleness_score", 0.0),
                reverse=True,
            )

        run_metadata = {
            **result.metadata,
            "source_image_id": source_image_id,
            "raw_candidate_count": len(result.masks),
            "postprocessed_candidate_count": len(candidates),
            "objectness_prior_enabled": self.prior_scorer is not None,
        }

        return CandidateBatch(
            source_image_id=source_image_id,
            image_rgb=result.image_rgb,
            candidates=candidates,
            run_metadata=run_metadata,
        )