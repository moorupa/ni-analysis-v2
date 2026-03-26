"""
Path
----
ni-analysis-v2/src/ni_analysis/export/reviewed_dataset_exporter.py

Role
----
Export reviewed masks and metadata into a reviewed dataset structure.

Export policy
-------------
- accepted masks are exported as primary reviewed assets
- edited masks are preferred over source masks when available
- metadata table is exported alongside mask files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_analysis.review.review_io import load_review_session
from ni_analysis.utils.io_utils import (
    build_reviewed_mask_path,
    ensure_dir,
    load_mask,
    save_json,
    save_mask,
)


def export_reviewed_dataset(
    review_session_path: str | Path,
    output_dir: str | Path,
    accepted_only: bool = True,
) -> dict:
    session = load_review_session(review_session_path)
    output_dir = ensure_dir(output_dir)

    exported_rows: list[dict] = []

    for decision in session.decisions:
        if accepted_only and decision.review_label != "accept_single":
            continue

        chosen_mask_path = decision.edited_mask_path or decision.source_mask_path
        if not chosen_mask_path:
            continue

        mask = load_mask(chosen_mask_path)
        save_path = build_reviewed_mask_path(output_dir, decision.candidate_id)
        save_mask(mask, save_path)

        exported_rows.append(
            {
                "session_id": session.session_id,
                "source_image_id": session.source_image_id,
                "source_image_path": session.source_image_path,
                "candidate_id": decision.candidate_id,
                "review_label": decision.review_label,
                "morphology_label": decision.morphology_label,
                "confidence": decision.confidence,
                "reviewer_id": decision.reviewer_id,
                "comment": decision.comment,
                "source_mask_path": decision.source_mask_path,
                "edited_mask_path": decision.edited_mask_path,
                "exported_mask_path": str(save_path),
                "bbox_xyxy": decision.bbox_xyxy,
                "extra_metadata": decision.extra_metadata,
            }
        )

    table_path = output_dir / "reviewed_dataset.csv"
    pd.DataFrame(exported_rows).to_csv(table_path, index=False, encoding="utf-8-sig")

    manifest = {
        "session_id": session.session_id,
        "source_image_id": session.source_image_id,
        "source_image_path": session.source_image_path,
        "accepted_only": accepted_only,
        "export_count": len(exported_rows),
        "table_path": str(table_path),
    }
    manifest_path = output_dir / "reviewed_dataset_manifest.json"
    save_json(manifest, manifest_path)

    return {
        "export_count": len(exported_rows),
        "table_path": str(table_path),
        "manifest_path": str(manifest_path),
    }