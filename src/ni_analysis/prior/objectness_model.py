from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = object


FEATURE_KEYS = [
    "raw_sam_score",
    "area_px",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "is_border_touching",
]


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def feature_vector_from_metadata(metadata: dict) -> np.ndarray:
    return np.array(
        [
            _safe_float(metadata.get("raw_sam_score"), 0.0),
            _safe_float(metadata.get("area_px"), 0.0),
            _safe_float(metadata.get("bbox_width"), 0.0),
            _safe_float(metadata.get("bbox_height"), 0.0),
            _safe_float(metadata.get("aspect_ratio"), 0.0),
            _safe_float(metadata.get("extent"), 0.0),
            _safe_float(metadata.get("solidity"), 0.0),
            _safe_float(metadata.get("circularity"), 0.0),
            1.0 if bool(metadata.get("is_border_touching", False)) else 0.0,
        ],
        dtype=np.float32,
    )


def review_label_to_target(label: str | None) -> int | None:
    if label is None:
        return None

    label = label.strip().lower()

    positive = {"accept", "accepted", "keep", "particle", "foreground"}
    negative = {
        "reject",
        "rejected",
        "artifact",
        "background",
        "fragment",
        "trash",
    }

    if label in positive:
        return 1
    if label in negative:
        return 0
    return None


@dataclass
class StandardizationStats:
    mean: list[float]
    std: list[float]

    @classmethod
    def from_array(cls, x: np.ndarray) -> "StandardizationStats":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.tolist(), std=std.tolist())

    def apply(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (x - mean) / std


class CandidateFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], stats: StandardizationStats | None = None) -> None:
        self.samples: list[tuple[np.ndarray, int]] = []

        for row in rows:
            y = review_label_to_target(row.get("review_label"))
            if y is None:
                continue

            feats = feature_vector_from_metadata(row)
            self.samples.append((feats, y))

        if not self.samples:
            raise ValueError("No labeled samples found for objectness training.")

        x = np.stack([s[0] for s in self.samples], axis=0)
        self.stats = stats or StandardizationStats.from_array(x)

        self.x = self.stats.apply(x).astype(np.float32)
        self.y = np.asarray([s[1] for s in self.samples], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if torch is None:
            raise RuntimeError("PyTorch is required for training objectness model.")
        return torch.from_numpy(self.x[idx]), torch.tensor([self.y[idx]], dtype=torch.float32)


class MetadataMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainedObjectnessBundle:
    model_state_path: Path
    stats_path: Path
    feature_keys: list[str]

    def save_stats(self, stats: StandardizationStats) -> None:
        payload = {
            "feature_keys": self.feature_keys,
            "mean": stats.mean,
            "std": stats.std,
        }
        self.stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class MetadataObjectnessScorer:
    def __init__(self, model_path: str | Path, stats_path: str | Path, device: str = "cpu") -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for objectness inference.")

        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path)
        self.device = device

        stats_payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.stats = StandardizationStats(
            mean=stats_payload["mean"],
            std=stats_payload["std"],
        )
        self.feature_keys = stats_payload["feature_keys"]

        self.model = MetadataMLP(in_dim=len(self.feature_keys))
        state = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    def score_metadata(self, metadata: dict) -> float:
        feats = feature_vector_from_metadata(metadata)[None, :]
        feats = self.stats.apply(feats).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(feats).to(self.device)
            logit = self.model(x)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)[0]
        return float(prob)

    def score_batch(self, candidates: list) -> list:
        for cand in candidates:
            score = self.score_metadata(cand.metadata)
            cand.metadata["particleness_score"] = float(score)
        return candidates


def train_metadata_objectness_model(
    rows: list[dict],
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainedObjectnessBundle:
    if torch is None:
        raise RuntimeError("PyTorch is required for training objectness model.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = CandidateFeatureDataset(rows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MetadataMLP(in_dim=len(FEATURE_KEYS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.shape[0]

        epoch_loss = running_loss / len(ds)
        print(f"[Epoch {epoch + 1:03d}] loss={epoch_loss:.6f}")

    model_path = output_dir / "metadata_objectness.pt"
    stats_path = output_dir / "metadata_objectness_stats.json"

    torch.save(model.state_dict(), model_path)

    bundle = TrainedObjectnessBundle(
        model_state_path=model_path,
        stats_path=stats_path,
        feature_keys=FEATURE_KEYS,
    )
    bundle.save_stats(ds.stats)
    return bundle