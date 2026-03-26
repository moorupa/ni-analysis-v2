"""
Path
----
ni-analysis-v2/src/ni_analysis/review/review_session_core.py

Role
----
Core manipulation logic for review sessions.

This module does not implement the full interactive UI yet.
Instead, it provides structured update helpers so the review workflow
can evolve without putting business logic directly into the script layer.
"""

from __future__ import annotations

from ni_analysis.review.review_schema import ReviewDecision, ReviewSessionRecord


def find_decision_index(session: ReviewSessionRecord, candidate_id: str) -> int:
    for idx, decision in enumerate(session.decisions):
        if decision.candidate_id == candidate_id:
            return idx
    raise KeyError(f"candidate_id not found in session: {candidate_id}")


def get_decision(session: ReviewSessionRecord, candidate_id: str) -> ReviewDecision:
    idx = find_decision_index(session, candidate_id)
    return session.decisions[idx]


def update_decision(
    session: ReviewSessionRecord,
    candidate_id: str,
    *,
    review_label: str | None = None,
    morphology_label: str | None = None,
    confidence: int | None = None,
    comment: str | None = None,
    edited_mask_path: str | None = None,
) -> None:
    idx = find_decision_index(session, candidate_id)
    decision = session.decisions[idx]

    if review_label is not None:
        decision.review_label = review_label  # type: ignore[assignment]
    if morphology_label is not None:
        decision.morphology_label = morphology_label  # type: ignore[assignment]
    if confidence is not None:
        decision.confidence = int(confidence)
    if comment is not None:
        decision.comment = comment
    if edited_mask_path is not None:
        decision.edited_mask_path = edited_mask_path


def summarize_session(session: ReviewSessionRecord) -> dict:
    summary = {
        "session_id": session.session_id,
        "source_image_id": session.source_image_id,
        "total": len(session.decisions),
        "accepted": 0,
        "rejected": 0,
        "needs_redraw": 0,
        "unreviewed_like": 0,
    }

    for d in session.decisions:
        if d.review_label == "accept_single":
            summary["accepted"] += 1
        elif d.review_label == "needs_redraw":
            summary["needs_redraw"] += 1
        elif d.review_label == "reject_uncertain":
            summary["unreviewed_like"] += 1
        else:
            summary["rejected"] += 1

    return summary