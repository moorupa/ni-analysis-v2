"""
Path
----
ni-analysis-v2/scripts/extract_features_from_reviewed.py

Role
----
Extract morphology features from reviewed/exported masks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.features.feature_table import (
    build_feature_dataframe,
    save_feature_outputs,
)
from ni_analysis.features.morphology_features import compute_all_shape_features
from ni_analysis.utils.io_utils import load_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract morphology features from reviewed dataset CSV."
    )
    parser.add_argument(
        "--reviewed-csv",
        type=str,
        required=True,
        help="Path to reviewed_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save feature tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reviewed_csv = Path(args.reviewed_csv)
    if not reviewed_csv.exists():
        raise FileNotFoundError(f"Reviewed CSV not found: {reviewed_csv}")

    reviewed_df = pd.read_csv(reviewed_csv)
    rows: list[dict] = []

    for _, item in reviewed_df.iterrows():
        mask_path = item.get("exported_mask_path")
        if pd.isna(mask_path) or not mask_path:
            continue

        mask = load_mask(mask_path)
        feats = compute_all_shape_features(mask)

        row = item.to_dict()
        row.update(feats)
        rows.append(row)

    feature_df = build_feature_dataframe(rows)
    saved = save_feature_outputs(
        df=feature_df,
        output_dir=args.output_dir,
        base_name="reviewed_features",
    )

    print("[DONE] Feature extraction completed.")
    for key, value in saved.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()