from __future__ import annotations

from ni_analysis.review.review_io import build_empty_review_session_from_manifest
from ni_analysis.review.review_session_core import summarize_session, update_decision


def _manifest() -> dict:
    return {
        "source_image_id": "img001",
        "source_image_path": "data/img001.png",
        "overlay_path": "runs/overlay.png",
        "run_metadata": {"mode": "test"},
        "candidates": [
            {
                "candidate_id": "img001_cand_0001",
                "bbox_xyxy": [1, 2, 10, 20],
                "mask_path": "masks/1.png",
                "metadata": {"raw_sam_score": 0.7},
            },
            {
                "candidate_id": "img001_cand_0002",
                "bbox_xyxy": [3, 4, 7, 8],
                "mask_path": "masks/2.png",
                "metadata": {},
            },
        ],
    }


def test_build_session_and_update_summary() -> None:
    session = build_empty_review_session_from_manifest(
        manifest=_manifest(),
        session_id="sess01",
        reviewer_id="reviewer",
    )

    summary = summarize_session(session)
    assert summary["total"] == 2
    assert summary["unreviewed_like"] == 2

    update_decision(
        session,
        candidate_id="img001_cand_0001",
        review_label="accept_single",
        morphology_label="spiky",
        confidence=5,
        comment="clear spikes",
    )

    summary2 = summarize_session(session)
    assert summary2["accepted"] == 1
    assert summary2["unreviewed_like"] == 1
