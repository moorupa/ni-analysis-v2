"""
Path
----
ni-analysis-v2/src/ni_analysis/utils/config_io.py

Role
----
Simple YAML config loader for ni-analysis-v2 scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}