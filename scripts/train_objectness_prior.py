from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ni_analysis.prior.objectness_model import train_metadata_objectness_model
from ni_analysis.utils.io_utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train metadata-based objectness prior from reviewed candidates."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.dataset)
    rows = payload.get("rows", [])

    bundle = train_metadata_objectness_model(
        rows=rows,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    print("[DONE] Objectness prior trained.")
    print(f"Model : {bundle.model_state_path}")
    print(f"Stats : {bundle.stats_path}")


if __name__ == "__main__":
    main()