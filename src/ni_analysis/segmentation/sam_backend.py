"""
Path
----
ni-analysis-v2/src/ni_analysis/segmentation/sam_backend.py

Role
----
Low-level backend wrapper around SAM/SAM-like models.

This module is intentionally prompt-agnostic for the main v2 path.
Its primary responsibility is:
1) load model
2) preprocess image
3) generate image-conditioned candidate masks
4) optionally support legacy text-prompt path for baseline only
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class CandidateGenerationResult:
    masks: list[np.ndarray]
    scores: list[float]
    image_rgb: np.ndarray
    metadata: dict[str, Any]


class SAMBackend:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "sam3",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_type = model_type
        self.device = device or self._auto_device()
        self.model = self._build_model()

    def _auto_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_model(self) -> Any:
        """
        Replace this body with actual SAM/SAM2/SAM3 loading logic.
        """
        return None

    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        if not isinstance(pil_img, Image.Image):
            raise TypeError("pil_img must be a PIL.Image.Image")

        rgb = pil_img.convert("RGB")
        cv_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)

    def generate_candidates(
        self,
        image: Image.Image,
        apply_preprocessing: bool = True,
        sampling_grid_size: int = 32,
        pred_iou_thresh: float = 0.70,
        stability_score_thresh: float = 0.70,
        min_mask_region_area: int = 0,
    ) -> CandidateGenerationResult:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        processed = image.convert("RGB")
        if apply_preprocessing:
            processed = self.preprocess_image(processed)

        image_rgb = np.array(processed)

        masks: list[np.ndarray] = []
        scores: list[float] = []

        metadata = {
            "mode": "prompt_free_candidate_generation",
            "sampling_grid_size": sampling_grid_size,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "min_mask_region_area": min_mask_region_area,
            "device": self.device,
            "model_type": self.model_type,
        }

        return CandidateGenerationResult(
            masks=masks,
            scores=scores,
            image_rgb=image_rgb,
            metadata=metadata,
        )

    def segment_from_text(
        self,
        image: Image.Image,
        prompt: str,
        apply_preprocessing: bool = True,
    ) -> CandidateGenerationResult:
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        processed = image.convert("RGB")
        if apply_preprocessing:
            processed = self.preprocess_image(processed)

        image_rgb = np.array(processed)

        masks: list[np.ndarray] = []
        scores: list[float] = []

        metadata = {
            "mode": "legacy_text_prompt_baseline",
            "prompt": prompt,
            "device": self.device,
            "model_type": self.model_type,
        }

        return CandidateGenerationResult(
            masks=masks,
            scores=scores,
            image_rgb=image_rgb,
            metadata=metadata,
        )