"""LoRA training. Validation chooses the checkpoint. The test split is not read."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import gc
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _parent() -> int:
    for length in (1024, 768, 512):
        print(f"training with max_length={length}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", "--max-length", str(length)],
            cwd=ROOT,
        )
        if completed.returncode == 0:
            return 0
        if completed.returncode != 42:
            return completed.returncode
        print(f"out of memory at max_length={length}; trying a shorter note", flush=True)
    print("training did not fit in memory at max_length 512", flush=True)
    return 1


def _child(max_length: int) -> int:
    import torch

    from genai.data.records import read_jsonl
    from genai.models.load import target_dir
    from genai.paths import CONFIG_DIR, PROCESSED_DIR
    from genai.training.finetune import train_lora

    if not target_dir().exists():
        raise SystemExit("Download the target model first: uv run python scripts/download_models.py")
    config = json.loads((CONFIG_DIR / "train.json").read_text())
    config["max_length"] = max_length
    torch.manual_seed(int(config["seed"]))
    try:
        summary = train_lora(
            read_jsonl(PROCESSED_DIR / "train.jsonl"),
            read_jsonl(PROCESSED_DIR / "validation.jsonl"),
            target_dir(),
            config,
        )
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" in text or "mps backend" in text:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(exc, flush=True)
            return 42
        raise
    print("best epoch", summary["best_epoch"], "val macro-F1", summary["best_val_macro_f1"], flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.child:
        raise SystemExit(_child(args.max_length))
    raise SystemExit(_parent())
