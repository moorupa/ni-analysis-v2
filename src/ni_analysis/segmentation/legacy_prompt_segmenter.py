"""
Path
----
ni-analysis-v2/src/ni_analysis/segmentation/legacy_prompt_segmenter.py

Role
----
Legacy prompt-dependent baseline wrapper.

This file exists to make it explicit that the prompt path is not the main
v2 workflow, but a comparison baseline only.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ni_analysis.segmentation.sam_backend import SAMBackend


class LegacyPromptSegmenter:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "sam3",
        device: str | None = None,
    ) -> None:
        self.backend = SAMBackend(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
        )

    def segment(
        self,
        image: Image.Image,
        prompt: str,
        apply_preprocessing: bool = True,
    ):
        return self.backend.segment_from_text(
            image=image,
            prompt=prompt,
            apply_preprocessing=apply_preprocessing,
        )