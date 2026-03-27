from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export candidate-level objectness dataset from review session + manifest."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def build_candidate_lookup(manifest: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}

    for item in manifest.get("candidates", []):
        cid = item["candidate_id"]
        lookup[cid] = item

    return lookup


def main() -> None:
    args = parse_args()

    manifest = load_json(args.manifest)
    session = load_json(args.session)

    candidate_lookup = build_candidate_lookup(manifest)

    rows: list[dict] = []
    decisions = session.get("decisions", {})

    for candidate_id, decision in decisions.items():
        cand = candidate_lookup.get(candidate_id)
        if cand is None:
            continue

        metadata = cand.get("metadata", {}).copy()
        bbox = cand.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox

        metadata["area_px"] = cand.get("area_px", metadata.get("area_px", 0))
        metadata["raw_sam_score"] = cand.get("score", metadata.get("raw_sam_score", 0.0))
        metadata["bbox_width"] = max(0, int(x2) - int(x1))
        metadata["bbox_height"] = max(0, int(y2) - int(y1))

        row = {
            "candidate_id": candidate_id,
            "review_label": decision.get("review_label"),
            "morphology_label": decision.get("morphology_label"),
            "confidence": decision.get("confidence"),
            "comment": decision.get("comment"),
            **metadata,
        }
        rows.append(row)

    output_payload = {
        "num_rows": len(rows),
        "rows": rows,
    }
    save_json(output_payload, args.output)

    print("[DONE] Objectness dataset exported.")
    print(f"Rows   : {len(rows)}")
    print(f"Output : {args.output}")


if __name__ == "__main__":
    main()