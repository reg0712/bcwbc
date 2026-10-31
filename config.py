"""Configuration files under config/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from synapse import ROOT


def load_config(name: str) -> dict[str, Any]:
    """Parse config/<name>, a YAML mapping."""
    return yaml.safe_load((ROOT / "config" / name).read_text()) or {}


def load_env(path: Path = ROOT / ".env") -> None:
    """Copy KEY=VALUE lines from .env into the environment. Variables already set win, and
    empty values are skipped, so LANGSMITH_TRACING= leaves tracing off."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
