from __future__ import annotations

import pandas as pd

from ni_analysis.features.feature_table import (
    build_feature_dataframe,
    summarize_by_morphology,
    summarize_numeric_features,
)


def test_feature_table_helpers_smoke() -> None:
    rows = [
        {
            "candidate_id": "a",
            "morphology_label": "spiky",
            "area_px": 10,
            "circularity": 0.4,
        },
        {
            "candidate_id": "b",
            "morphology_label": "smooth",
            "area_px": 12,
            "circularity": 0.8,
        },
    ]

    df = build_feature_dataframe(rows)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    numeric = summarize_numeric_features(df, feature_columns=["area_px", "circularity"])
    by_morph = summarize_by_morphology(df, feature_columns=["area_px", "circularity"])

    assert not numeric.empty
    assert not by_morph.empty
