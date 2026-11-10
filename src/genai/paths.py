"""Repository paths. Everything is relative to the Ass2 folder."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = SPLIT_DIR / "manifest.json"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
ADAPTER_DIR = OUTPUTS_DIR / "task1" / "adapter"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "configs"
