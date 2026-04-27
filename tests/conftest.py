from __future__ import annotations

import sys
import types


try:
    import cv2  # noqa: F401
except Exception:
    fake_cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = fake_cv2
