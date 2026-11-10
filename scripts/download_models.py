"""Download the Qwen2.5 target and draft weights into models/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download

from genai.constants import DRAFT_MODEL_ID, TARGET_MODEL_ID
from genai.models.load import draft_dir, target_dir

ALLOW = ["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "LICENSE*", "README*"]


def download(model_id: str, path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=path, allow_patterns=ALLOW)
    print("saved", model_id, "to", path)


if __name__ == "__main__":
    download(TARGET_MODEL_ID, target_dir())
    download(DRAFT_MODEL_ID, draft_dir())
