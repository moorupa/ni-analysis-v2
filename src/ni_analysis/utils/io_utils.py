"""
Path
----
ni-analysis-v2/src/ni_analysis/utils/io_utils.py

Role
----
Workflow-oriented I/O utilities for ni-analysis-v2.

This module is designed around the v2 pipeline:
candidate generation -> review -> reviewed dataset export

It intentionally goes beyond generic image saving helpers and provides
consistent file naming / manifest saving primitives for the workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got shape={arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return Path(path)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_rgb_image(image: np.ndarray, path: str | Path) -> Path:
    """
    Save RGB image to disk.
    """
    path = ensure_parent(path)
    img = to_uint8_image(image)

    if img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return Path(path)


def load_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    path = ensure_parent(path)
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask_u8)
    return Path(path)


def load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def build_candidate_mask_path(
    output_dir: str | Path,
    candidate_id: str,
) -> Path:
    return Path(output_dir) / "candidate_masks" / f"{candidate_id}.png"


def build_reviewed_mask_path(
    output_dir: str | Path,
    candidate_id: str,
) -> Path:
    return Path(output_dir) / "reviewed_masks" / f"{candidate_id}.png"


def overlay_masks_on_image(
    image: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.40,
    draw_boundaries: bool = True,
) -> np.ndarray:
    """
    Overlay binary masks on an RGB image with deterministic pseudo-colors.
    """
    base = to_uint8_image(image).copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    overlay = base.copy()

    for idx, mask in enumerate(masks):
        mask_bin = (np.asarray(mask) > 0).astype(np.uint8)

        color = np.array([
            (37 * (idx + 1)) % 255,
            (97 * (idx + 1)) % 255,
            (173 * (idx + 1)) % 255,
        ], dtype=np.uint8)

        overlay[mask_bin > 0] = (
            (1.0 - alpha) * overlay[mask_bin > 0] + alpha * color
        ).astype(np.uint8)

        if draw_boundaries:
            contours, _ = cv2.findContours(
                mask_bin,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay


def save_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    path: str | Path,
    alpha: float = 0.40,
    draw_boundaries: bool = True,
) -> Path:
    overlay = overlay_masks_on_image(
        image=image,
        masks=masks,
        alpha=alpha,
        draw_boundaries=draw_boundaries,
    )
    return save_rgb_image(overlay, path)


def build_candidate_manifest(
    source_image_id: str,
    source_image_path: str,
    overlay_path: str,
    run_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_image_id": source_image_id,
        "source_image_path": source_image_path,
        "overlay_path": overlay_path,
        "candidate_count": len(candidates),
        "run_metadata": run_metadata,
        "candidates": candidates,
    }


def save_candidate_manifest(
    source_image_id: str,
    source_image_path: str,
    overlay_path: str,
    run_metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    manifest = build_candidate_manifest(
        source_image_id=source_image_id,
        source_image_path=source_image_path,
        overlay_path=overlay_path,
        run_metadata=run_metadata,
        candidates=candidates,
    )
    return save_json(manifest, path)