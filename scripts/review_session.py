"""
review_session.py

Role
----
CLI entry point for a review session.

Current skeleton scope
----------------------
- load candidate manifest
- initialize empty review decisions
- save a review-session JSON scaffold

Future expansion
----------------
- OpenCV/Qt interactive review UI
- keyboard label shortcuts
- local redraw / box-refine integration
- edited mask save-back
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.review.review_schema import ReviewDecision, ReviewSessionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize or continue a review session.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to candidate_manifest.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for review outputs")
    parser.add_argument("--session-id", type=str, required=True, help="Review session ID")
    parser.add_argument("--reviewer-id", type=str, default="default_reviewer", help="Reviewer ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    decisions: list[ReviewDecision] = []
    for item in manifest.get("candidates", []):
        decisions.append(
            ReviewDecision(
                candidate_id=item["candidate_id"],
                review_label="reject_uncertain",
                morphology_label="unlabeled",
                confidence=3,
                reviewer_id=args.reviewer_id,
                source_mask_path=item.get("mask_path"),
                source_image_path=manifest.get("source_image_path"),
                bbox_xyxy=tuple(item["bbox_xyxy"]) if item.get("bbox_xyxy") is not None else None,
                comment="initialized; not yet reviewed",
            )
        )

    session = ReviewSessionRecord(
        session_id=args.session_id,
        source_image_id=manifest["source_image_id"],
        source_image_path=manifest["source_image_path"],
        decisions=decisions,
        session_metadata={
            "manifest_path": str(manifest_path),
            "candidate_count": len(decisions),
            "status": "initialized",
        },
    )

    session_path = output_dir / f"{args.session_id}.review.json"
    with session_path.open("w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    print("[DONE] Review session scaffold created.")
    print(f"Session file: {session_path}")


if __name__ == "__main__":
    main()