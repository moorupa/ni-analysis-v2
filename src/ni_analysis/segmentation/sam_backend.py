"""
Path
----
ni-analysis-v2/src/ni_analysis/segmentation/sam_backend.py

Role
----
Low-level backend wrapper around SAM/SAM-like models.

Main v2 path
------------
- prompt-free candidate generation via automatic mask generation

Legacy path
-----------
- text-prompt baseline is preserved separately and is not the main workflow
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
        Build model object for the selected backend.

        Current behavior
        ----------------
        - returns None if the actual SAM backend is not available yet
        - this lets the repository structure work before the final model binding

        Replace the branch bodies below with your actual SAM/SAM2/SAM3 imports.
        """
        if self.model_type.lower() in {"sam3", "sam2", "sam"}:
            # Example future implementation sketch:
            #
            # if self.model_type.lower() == "sam3":
            #     from sam3.model_builder import build_sam3_image_model
            #     model = build_sam3_image_model(checkpoint=str(self.checkpoint_path))
            # elif self.model_type.lower() == "sam2":
            #     from sam2.build_sam import build_sam2
            #     model = build_sam2(checkpoint=str(self.checkpoint_path))
            # else:
            #     from segment_anything import sam_model_registry
            #     model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
            #
            # model.to(self.device)
            # model.eval()
            # return model
            return None

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        """
        CLAHE-style contrast enhancement useful for SEM-like grayscale textures.
        """
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

    def _generate_candidates_with_backend(
        self,
        image_rgb: np.ndarray,
        sampling_grid_size: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        min_mask_region_area: int,
    ) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
        """
        Automatic candidate generation using the selected SAM-like backend.

        This is the method you should replace first with your real backend call.
        """
        if self.model is None:
            return self._generate_candidates_fallback(
                image_rgb=image_rgb,
                sampling_grid_size=sampling_grid_size,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )

        model_name = self.model_type.lower()

        if model_name in {"sam", "sam2", "sam3"}:
            # Replace this block with your actual automatic mask generator.
            #
            # Example conceptual pattern:
            #   generator = SamAutomaticMaskGenerator(
            #       model=self.model,
            #       points_per_side=sampling_grid_size,
            #       pred_iou_thresh=pred_iou_thresh,
            #       stability_score_thresh=stability_score_thresh,
            #       min_mask_region_area=min_mask_region_area,
            #   )
            #   raw = generator.generate(image_rgb)
            #   masks = [(item["segmentation"] > 0).astype(np.uint8) for item in raw]
            #   scores = [float(item.get("predicted_iou", 0.0)) for item in raw]
            #
            #   return masks, scores, {"backend_mode": "automatic_mask_generator"}
            return self._generate_candidates_fallback(
                image_rgb=image_rgb,
                sampling_grid_size=sampling_grid_size,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )

        raise ValueError(f"Unsupported model_type for candidate generation: {self.model_type}")

    def _generate_candidates_fallback(
        self,
        image_rgb: np.ndarray,
        sampling_grid_size: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        min_mask_region_area: int,
    ) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
        """
        Temporary fallback candidate generator.

        Purpose
        -------
        Keeps the repo executable before the real SAM automatic-mask path is wired.

        Behavior
        --------
        - grayscale conversion
        - Otsu thresholding
        - connected-component proposals
        - very rough heuristic scores

        This is NOT the final scientific method.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Heuristic: invert if foreground coverage is absurdly high
        foreground_ratio = (thresh > 0).mean()
        if foreground_ratio > 0.65:
            thresh = cv2.bitwise_not(thresh)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (thresh > 0).astype(np.uint8), connectivity=8
        )

        masks: list[np.ndarray] = []
        scores: list[float] = []

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < max(1, int(min_mask_region_area)):
                continue

            component = (labels == label_idx).astype(np.uint8)
            masks.append(component)

            # crude placeholder score: normalized by image area
            norm_score = min(1.0, area / max(1.0, image_rgb.shape[0] * image_rgb.shape[1] * 0.1))
            scores.append(float(norm_score))

        metadata = {
            "backend_mode": "fallback_connected_components",
            "foreground_ratio_after_threshold": float(foreground_ratio),
            "sampling_grid_size": sampling_grid_size,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "min_mask_region_area": min_mask_region_area,
        }
        return masks, scores, metadata

    def generate_candidates(
        self,
        image: Image.Image,
        apply_preprocessing: bool = True,
        sampling_grid_size: int = 32,
        pred_iou_thresh: float = 0.70,
        stability_score_thresh: float = 0.70,
        min_mask_region_area: int = 0,
    ) -> CandidateGenerationResult:
        """
        Main prompt-free entry for candidate generation.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        processed = image.convert("RGB")
        if apply_preprocessing:
            processed = self.preprocess_image(processed)

        image_rgb = np.array(processed)

        masks, scores, backend_metadata = self._generate_candidates_with_backend(
            image_rgb=image_rgb,
            sampling_grid_size=sampling_grid_size,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

        metadata = {
            "mode": "prompt_free_candidate_generation",
            "device": self.device,
            "model_type": self.model_type,
            "apply_preprocessing": apply_preprocessing,
            **backend_metadata,
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
        """
        Legacy baseline path only.
        """
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        processed = image.convert("RGB")
        if apply_preprocessing:
            processed = self.preprocess_image(processed)

        image_rgb = np.array(processed)

        # Placeholder until legacy prompt path is bound to real backend
        masks: list[np.ndarray] = []
        scores: list[float] = []

        metadata = {
            "mode": "legacy_text_prompt_baseline",
            "prompt": prompt,
            "device": self.device,
            "model_type": self.model_type,
            "backend_mode": "placeholder",
        }

        return CandidateGenerationResult(
            masks=masks,
            scores=scores,
            image_rgb=image_rgb,
            metadata=metadata,
        )