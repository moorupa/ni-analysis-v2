"""
Path
----
ni-analysis-v2/src/ni_analysis/features/spikiness_features.py

Role
----
Experimental spikiness-oriented descriptors for binary particle masks.
"""

from __future__ import annotations

import numpy as np
from skimage import measure
from skimage.morphology import convex_hull_image


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    mask = (mask > 0).astype(np.uint8)
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return None
    return max(contours, key=lambda c: c.shape[0])


def convexity_deficit(mask: np.ndarray) -> float:
    mask = (mask > 0).astype(np.uint8)
    area = float(mask.sum())
    if area <= 0:
        return float("nan")

    hull = convex_hull_image(mask > 0)
    hull_area = float(hull.sum())
    if hull_area <= 0:
        return float("nan")

    return (hull_area - area) / hull_area


def peak_count_proxy(mask: np.ndarray, smooth_window: int = 9) -> float:
    contour = _largest_contour(mask)
    if contour is None or len(contour) < smooth_window + 2:
        return float("nan")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan")

    cx = float(xs.mean())
    cy = float(ys.mean())

    dists = np.sqrt((contour[:, 1] - cx) ** 2 + (contour[:, 0] - cy) ** 2)

    kernel = np.ones(smooth_window) / smooth_window
    smooth = np.convolve(dists, kernel, mode="same")
    residual = dists - smooth

    threshold = np.std(residual)
    peaks = residual > threshold
    return float(np.sum(peaks))


def compute_spikiness_features(mask: np.ndarray) -> dict[str, float]:
    return {
        "convexity_deficit": convexity_deficit(mask),
        "peak_count_proxy": peak_count_proxy(mask),
    }