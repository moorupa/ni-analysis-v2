"""
Path
----
ni-analysis-v2/src/ni_analysis/review/review_io.py

Role
----
Load/save helpers for review session files.

Design note
-----------
Schema definition and serialization are separated so that:
- review_schema.py stays focused on typed structures
- review_io.py handles persistence concerns
"""

from __future__ import annotations

from pathlib import Path

from ni_analysis.review.review_schema import (
    ReviewDecision,
    ReviewSessionRecord,
)
from ni_analysis.utils.io_utils import load_json, save_json


def decision_from_dict(data: dict) -> ReviewDecision:
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
        source_mask_path=data.get("source_mask_path"),
        edited_mask_path=data.get("edited_mask_path"),
        source_image_path=data.get("source_image_path"),
        bbox_xyxy=bbox,
        extra_metadata=data.get("extra_metadata", {}),
    )


def session_from_dict(data: dict) -> ReviewSessionRecord:
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
    manifest: dict,
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
                source_mask_path=item.get("mask_path"),
                edited_mask_path=None,
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
            "overlay_path": manifest.get("overlay_path"),
            "run_metadata": manifest.get("run_metadata", {}),
        },
    )