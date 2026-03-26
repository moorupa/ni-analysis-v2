"""
Path
----
ni-analysis-v2/src/ni_analysis/features/roughness_features.py

Role
----
Experimental roughness-related descriptors for binary particle masks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    mask = (mask > 0).astype(np.uint8)
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return None
    return max(contours, key=lambda c: c.shape[0])


def perimeter_to_hull_perimeter_ratio(mask: np.ndarray) -> float:
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 3:
        return float("nan")

    perimeter = float(np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1)))

    try:
        hull = ConvexHull(contour)
        hull_pts = contour[hull.vertices]
        hull_closed = np.vstack([hull_pts, hull_pts[0]])
        hull_perimeter = float(
            np.sum(np.linalg.norm(np.diff(hull_closed, axis=0), axis=1))
        )
    except Exception:
        return float("nan")

    if hull_perimeter <= 0:
        return float("nan")
    return perimeter / hull_perimeter


def boundary_radial_std(mask: np.ndarray) -> float:
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 3:
        return float("nan")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan")

    cx = float(xs.mean())
    cy = float(ys.mean())

    dists = np.sqrt((contour[:, 1] - cx) ** 2 + (contour[:, 0] - cy) ** 2)
    if len(dists) == 0:
        return float("nan")
    return float(np.std(dists))


def compute_roughness_features(mask: np.ndarray) -> dict[str, float]:
    return {
        "perimeter_to_hull_perimeter_ratio": perimeter_to_hull_perimeter_ratio(mask),
        "boundary_radial_std": boundary_radial_std(mask),
    }