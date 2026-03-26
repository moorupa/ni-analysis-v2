"""
Path
----
ni-analysis-v2/scripts/review_session.py

Role
----
CLI entry point for initializing or inspecting a review session.

Current scope
-------------
1) initialize a review session from candidate_manifest.json
2) load an existing review session
3) print session summary
4) save scaffold JSON

Future scope
------------
- keyboard-driven review UI
- OpenCV overlay visualization
- edited mask save-back
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.review.review_io import (
    build_empty_review_session_from_manifest,
    load_review_session,
    save_review_session,
)
from ni_analysis.review.review_session_core import summarize_session
from ni_analysis.utils.io_utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize or inspect review session.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize review session from manifest")
    init_parser.add_argument("--manifest", type=str, required=True)
    init_parser.add_argument("--output-dir", type=str, required=True)
    init_parser.add_argument("--session-id", type=str, required=True)
    init_parser.add_argument("--reviewer-id", type=str, default="default_reviewer")

    summary_parser = subparsers.add_parser("summary", help="Print review session summary")
    summary_parser.add_argument("--session", type=str, required=True)

    return parser.parse_args()


def cmd_init(args: argparse.Namespace) -> None:
    manifest = load_json(args.manifest)
    output_dir = ensure_dir(args.output_dir)

    session = build_empty_review_session_from_manifest(
        manifest=manifest,
        session_id=args.session_id,
        reviewer_id=args.reviewer_id,
    )

    session_path = output_dir / f"{args.session_id}.review.json"
    save_review_session(session, session_path)

    summary = summarize_session(session)

    print("[DONE] Review session initialized.")
    print(f"Session path : {session_path}")
    print(f"Total        : {summary['total']}")
    print(f"Accepted     : {summary['accepted']}")
    print(f"Rejected     : {summary['rejected']}")
    print(f"Needs redraw : {summary['needs_redraw']}")
    print(f"Unreviewed   : {summary['unreviewed_like']}")


def cmd_summary(args: argparse.Namespace) -> None:
    session = load_review_session(args.session)
    summary = summarize_session(session)

    print("[INFO] Review session summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    args = parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()