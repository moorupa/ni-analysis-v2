"""
Path
----
ni-analysis-v2/src/ni_analysis/features/feature_table.py

Role
----
Utilities for building and summarizing review-aware feature tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "area_px",
    "perimeter_px",
    "equivalent_diameter_px",
    "aspect_ratio",
    "extent",
    "convex_area_px",
    "solidity",
    "circularity",
]


def build_feature_dataframe(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df

    preferred_front = [
        "session_id",
        "source_image_id",
        "source_image_path",
        "candidate_id",
        "review_label",
        "morphology_label",
        "confidence",
        "reviewer_id",
        "comment",
        "source_mask_path",
        "edited_mask_path",
        "exported_mask_path",
    ]

    ordered_cols = [c for c in preferred_front if c in df.columns]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols]


def summarize_numeric_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    valid_cols = [c for c in feature_columns if c in df.columns]

    if not valid_cols:
        return pd.DataFrame()

    return df[valid_cols].describe().T


def summarize_by_morphology(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty or "morphology_label" not in df.columns:
        return pd.DataFrame()

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    valid_cols = [c for c in feature_columns if c in df.columns]

    if not valid_cols:
        return pd.DataFrame()

    return df.groupby("morphology_label")[valid_cols].mean(numeric_only=True)


def save_feature_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
    base_name: str = "reviewed_features",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = output_dir / f"{base_name}.csv"
    summary_csv = output_dir / f"{base_name}_summary.csv"
    morphology_csv = output_dir / f"{base_name}_by_morphology.csv"

    df.to_csv(feature_csv, index=False, encoding="utf-8-sig")

    summary_df = summarize_numeric_features(df)
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, encoding="utf-8-sig")

    morphology_df = summarize_by_morphology(df)
    if not morphology_df.empty:
        morphology_df.to_csv(morphology_csv, encoding="utf-8-sig")

    return {
        "feature_csv": str(feature_csv),
        "summary_csv": str(summary_csv),
        "morphology_csv": str(morphology_csv),
    }