"""
Path
----
ni-analysis-v2/src/ni_analysis/features/morphology_features.py

Role
----
Basic mask-based morphology feature extraction for ni-analysis-v2.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ni_analysis.features.roughness_features import compute_roughness_features
from ni_analysis.features.spikiness_features import compute_spikiness_features


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={arr.shape}")
    return (arr > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = to_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:
        return m

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(component_areas))
    return (labels == largest_idx).astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    m = keep_largest_component(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    m = keep_largest_component(mask)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)


def compute_morphology_features(mask: np.ndarray) -> dict[str, Any]:
    m = keep_largest_component(mask)
    area = int(m.sum())

    if area == 0:
        return {
            "area_px": 0,
            "perimeter_px": float("nan"),
            "equivalent_diameter_px": float("nan"),
            "bbox_x": None,
            "bbox_y": None,
            "bbox_w": None,
            "bbox_h": None,
            "aspect_ratio": float("nan"),
            "extent": float("nan"),
            "convex_area_px": float("nan"),
            "solidity": float("nan"),
            "circularity": float("nan"),
        }

    contour = contour_from_mask(m)
    if contour is None:
        raise RuntimeError("Failed to extract contour from non-empty mask")

    perimeter = float(cv2.arcLength(contour, closed=True))
    eq_diameter = float(math.sqrt(4.0 * area / math.pi))

    bbox = bbox_xywh(m)
    if bbox is None:
        raise RuntimeError("Failed to compute bbox from non-empty mask")

    x, y, w, h = bbox
    bbox_area = float(w * h)
    aspect_ratio = float(w / h) if h > 0 else float("nan")
    extent = float(area / bbox_area) if bbox_area > 0 else float("nan")

    hull = cv2.convexHull(contour)
    convex_area = float(cv2.contourArea(hull))
    solidity = float(area / convex_area) if convex_area > 0 else float("nan")

    circularity = (
        float((4.0 * math.pi * area) / (perimeter ** 2))
        if perimeter > 0
        else float("nan")
    )

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "equivalent_diameter_px": eq_diameter,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "convex_area_px": convex_area,
        "solidity": solidity,
        "circularity": circularity,
    }


def compute_all_shape_features(mask: np.ndarray) -> dict[str, Any]:
    base = compute_morphology_features(mask)
    rough = compute_roughness_features(mask)
    spike = compute_spikiness_features(mask)
    return {**base, **rough, **spike}