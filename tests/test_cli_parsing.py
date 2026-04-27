from __future__ import annotations

from unittest.mock import patch

from scripts import export_objectness_dataset, review_session


def test_review_session_update_parse_args() -> None:
    argv = [
        "prog",
        "update",
        "--session",
        "runs/s.review.json",
        "--candidate-id",
        "img_cand_0001",
        "--review-label",
        "accept_single",
    ]
    with patch("sys.argv", argv):
        args = review_session.parse_args()

    assert args.command == "update"
    assert args.review_label == "accept_single"


def test_export_objectness_parse_args() -> None:
    argv = [
        "prog",
        "--manifest",
        "cand.json",
        "--session",
        "sess.json",
        "--output",
        "obj.json",
    ]
    with patch("sys.argv", argv):
        args = export_objectness_dataset.parse_args()

    assert args.manifest == "cand.json"
    assert args.session == "sess.json"
    assert args.output == "obj.json"


def test_iter_decisions_handles_list_format() -> None:
    session = {
        "decisions": [
            {"candidate_id": "a", "review_label": "accept_single"},
            {"candidate_id": "b", "review_label": "reject_noise"},
        ]
    }

    pairs = list(export_objectness_dataset.iter_decisions(session))
    assert pairs == [
        ("a", {"candidate_id": "a", "review_label": "accept_single"}),
        ("b", {"candidate_id": "b", "review_label": "reject_noise"}),
    ]
