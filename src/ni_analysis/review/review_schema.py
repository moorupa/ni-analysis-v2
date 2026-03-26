"""
Path
----
ni-analysis-v2/src/ni_analysis/review/review_schema.py

Role
----
Typed schema definitions for human review in ni-analysis-v2.

Why this exists
---------------
The reviewed output is not just a temporary annotation artifact.
It becomes:

1) the source of truth for accepted masks
2) the basis for reviewed dataset export
3) the basis for downstream feature extraction
4) a possible finetuning / hard-negative dataset later
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ReviewLabel = Literal[
    "accept_single",
    "reject_noise",
    "reject_artifact",
    "reject_border",
    "reject_merged",
    "reject_duplicate",
    "reject_uncertain",
    "needs_redraw",
]

MorphologyLabel = Literal[
    "spiky",
    "rough",
    "smooth",
    "agglomerated",
    "ambiguous",
    "unlabeled",
]


@dataclass
class ReviewDecision:
    candidate_id: str
    review_label: ReviewLabel
    morphology_label: MorphologyLabel = "unlabeled"
    confidence: int = 3  # 1~5
    reviewer_id: str = "default_reviewer"
    comment: str = ""
    source_mask_path: str | None = None
    edited_mask_path: str | None = None
    source_image_path: str | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewSessionRecord:
    session_id: str
    source_image_id: str
    source_image_path: str
    decisions: list[ReviewDecision]
    session_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_image_id": self.source_image_id,
            "source_image_path": self.source_image_path,
            "decisions": [d.to_dict() for d in self.decisions],
            "session_metadata": self.session_metadata,
        }


def validate_review_decision(decision: ReviewDecision) -> None:
    if not decision.candidate_id.strip():
        raise ValueError("candidate_id must be non-empty")

    if decision.confidence < 1 or decision.confidence > 5:
        raise ValueError("confidence must be between 1 and 5")