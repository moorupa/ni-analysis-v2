"""
Path
----
ni-analysis-v2/scripts/export_reviewed_dataset.py

Role
----
CLI entry point for exporting reviewed dataset assets from a review session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.export.reviewed_dataset_exporter import export_reviewed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed dataset from review session.")
    parser.add_argument("--session", type=str, required=True, help="Path to *.review.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Export all reviewed items instead of only accepted ones",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = export_reviewed_dataset(
        review_session_path=args.session,
        output_dir=args.output_dir,
        accepted_only=not args.include_rejected,
    )

    print("[DONE] Reviewed dataset export completed.")
    print(f"Export count : {result['export_count']}")
    print(f"CSV          : {result['table_path']}")
    print(f"Manifest     : {result['manifest_path']}")


if __name__ == "__main__":
    main()