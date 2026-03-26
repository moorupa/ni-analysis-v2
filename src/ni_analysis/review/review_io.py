"""
review_io.py

Role
----
Utilities for loading/saving review manifests and review session records.

This layer keeps JSON serialization logic separate from the review schema
definitions, so the schema can remain clean and typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ni_analysis.review.review_schema import (
    ReviewDecision,
    ReviewSessionRecord,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decision_from_dict(data: dict[str, Any]) -> ReviewDecision:
    bbox = data.get("bbox_xyxy")
    if bbox is not None:
        bbox = tuple(bbox)

    return ReviewDecision(
        candidate_id=data["candidate_id"],
        review_label=data["review_label"],
        morphology_label=data.get("morphology_label", "unlabeled"),
        confidence=int(data.get("confidence", 3)),
        reviewer_id=data.get("reviewer_id", "default_reviewer"),
        comment=data.get("comment", ""),
        edited_mask_path=data.get("edited_mask_path"),
        source_mask_path=data.get("source_mask_path"),
        source_image_path=data.get("source_image_path"),
        bbox_xyxy=bbox,
        extra_metadata=data.get("extra_metadata", {}),
    )


def session_from_dict(data: dict[str, Any]) -> ReviewSessionRecord:
    decisions = [decision_from_dict(d) for d in data.get("decisions", [])]
    return ReviewSessionRecord(
        session_id=data["session_id"],
        source_image_id=data["source_image_id"],
        source_image_path=data["source_image_path"],
        decisions=decisions,
        session_metadata=data.get("session_metadata", {}),
    )


def save_review_session(session: ReviewSessionRecord, path: str | Path) -> Path:
    return save_json(session.to_dict(), path)


def load_review_session(path: str | Path) -> ReviewSessionRecord:
    data = load_json(path)
    return session_from_dict(data)


def build_empty_review_session_from_manifest(
    manifest: dict[str, Any],
    session_id: str,
    reviewer_id: str = "default_reviewer",
) -> ReviewSessionRecord:
    decisions: list[ReviewDecision] = []

    for item in manifest.get("candidates", []):
        bbox = item.get("bbox_xyxy")
        if bbox is not None:
            bbox = tuple(bbox)

        decisions.append(
            ReviewDecision(
                candidate_id=item["candidate_id"],
                review_label="reject_uncertain",
                morphology_label="unlabeled",
                confidence=3,
                reviewer_id=reviewer_id,
                comment="initialized; pending review",
                edited_mask_path=None,
                source_mask_path=item.get("mask_path"),
                source_image_path=manifest.get("source_image_path"),
                bbox_xyxy=bbox,
                extra_metadata=item.get("metadata", {}),
            )
        )

    return ReviewSessionRecord(
        session_id=session_id,
        source_image_id=manifest["source_image_id"],
        source_image_path=manifest["source_image_path"],
        decisions=decisions,
        session_metadata={
            "candidate_count": len(decisions),
            "status": "initialized",
        },
    )