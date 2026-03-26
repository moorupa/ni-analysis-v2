"""
Path
----
ni-analysis-v2/src/ni_analysis/segmentation/mask_postprocess.py

Role
----
Postprocessing utilities for prompt-free candidate masks in ni-analysis-v2.

Design goal
-----------
This module is not a thin cleaning helper copied from v1.
It is designed for the v2 workflow:

raw candidates -> normalization -> cleanup -> de-duplication
-> nested-mask suppression -> review-ready ordering

Typical use
-----------
Use `postprocess_candidate_masks(...)` as the main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class PostprocessConfig:
    min_area: int = 50
    max_area: int | None = None
    kernel_size: int = 3
    do_open: bool = True
    do_close: bool = True
    fill_holes: bool = True
    remove_border_touching: bool = True
    deduplicate_iou_thresh: float = 0.85
    remove_nested: bool = True
    nested_overlap_thresh: float = 0.95
    sort_by: str = "area_desc"  # area_desc | area_asc | none


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary mask-like input to a binary uint8 mask in {0,1}.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={arr.shape}")
    return (arr > 0).astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(to_binary_mask(mask).sum())


def touches_border(mask: np.ndarray) -> bool:
    m = to_binary_mask(mask)
    return bool(
        m[0, :].any()
        or m[-1, :].any()
        or m[:, 0].any()
        or m[:, -1].any()
    )


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    m = to_binary_mask(mask).astype(bool)
    filled = ndimage.binary_fill_holes(m)
    return filled.astype(np.uint8)


def morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 3,
    do_open: bool = True,
    do_close: bool = True,
) -> np.ndarray:
    m = to_binary_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if do_open:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if do_close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    return (m > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = to_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:
        return m

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(component_areas))
    return (labels == largest_idx).astype(np.uint8)


def bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    m = to_binary_mask(mask)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def intersection_over_union(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = to_binary_mask(mask_a).astype(bool)
    b = to_binary_mask(mask_b).astype(bool)

    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def overlap_ratio(smaller_mask: np.ndarray, larger_mask: np.ndarray) -> float:
    """
    Ratio of overlap relative to the smaller mask area.
    Useful for nested-mask suppression.
    """
    s = to_binary_mask(smaller_mask).astype(bool)
    l = to_binary_mask(larger_mask).astype(bool)

    denom = s.sum()
    if denom == 0:
        return 0.0

    inter = np.logical_and(s, l).sum()
    return float(inter / denom)


def is_valid_by_area(mask: np.ndarray, min_area: int, max_area: int | None) -> bool:
    area = mask_area(mask)
    if area < min_area:
        return False
    if max_area is not None and area > max_area:
        return False
    return True


def deduplicate_masks(
    masks: list[np.ndarray],
    iou_thresh: float = 0.85,
) -> list[np.ndarray]:
    """
    Keep one representative mask among highly overlapping duplicates.
    Current strategy: keep earlier masks.
    """
    kept: list[np.ndarray] = []

    for candidate in masks:
        duplicate = False
        for prev in kept:
            if intersection_over_union(candidate, prev) >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


def remove_nested_masks(
    masks: list[np.ndarray],
    nested_overlap_thresh: float = 0.95,
) -> list[np.ndarray]:
    """
    Remove masks that are almost entirely contained inside another mask.
    Preference: keep the larger mask.
    """
    if not masks:
        return []

    indexed = [(idx, to_binary_mask(m), mask_area(m)) for idx, m in enumerate(masks)]
    indexed.sort(key=lambda x: x[2], reverse=True)  # larger first

    kept: list[np.ndarray] = []

    for _, current_mask, _ in indexed:
        nested = False
        for prev in kept:
            if overlap_ratio(current_mask, prev) >= nested_overlap_thresh:
                nested = True
                break
        if not nested:
            kept.append(current_mask)

    return kept


def sort_masks(
    masks: list[np.ndarray],
    mode: str = "area_desc",
) -> list[np.ndarray]:
    if mode == "none":
        return list(masks)

    reverse = mode == "area_desc"
    return sorted(masks, key=mask_area, reverse=reverse)


def clean_single_mask(
    mask: np.ndarray,
    config: PostprocessConfig,
) -> np.ndarray | None:
    """
    Normalize and clean a single candidate mask.
    Return None if rejected.
    """
    m = to_binary_mask(mask)

    if config.fill_holes:
        m = fill_mask_holes(m)

    m = morphological_cleanup(
        m,
        kernel_size=config.kernel_size,
        do_open=config.do_open,
        do_close=config.do_close,
    )

    m = keep_largest_component(m)

    if config.remove_border_touching and touches_border(m):
        return None

    if not is_valid_by_area(m, config.min_area, config.max_area):
        return None

    return m


def postprocess_candidate_masks(
    masks: Iterable[np.ndarray],
    config: PostprocessConfig | None = None,
) -> list[np.ndarray]:
    """
    Main entry point for v2 candidate-mask postprocessing.

    Steps
    -----
    1) clean each mask independently
    2) remove duplicates
    3) remove nested masks
    4) sort for review convenience
    """
    config = config or PostprocessConfig()

    cleaned: list[np.ndarray] = []
    for mask in masks:
        result = clean_single_mask(mask, config=config)
        if result is not None:
            cleaned.append(result)

    cleaned = deduplicate_masks(
        cleaned,
        iou_thresh=config.deduplicate_iou_thresh,
    )

    if config.remove_nested:
        cleaned = remove_nested_masks(
            cleaned,
            nested_overlap_thresh=config.nested_overlap_thresh,
        )

    cleaned = sort_masks(cleaned, mode=config.sort_by)
    return cleaned