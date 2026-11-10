"""Download the dataset and write data/splits/manifest.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.data.prepare import prepare


if __name__ == "__main__":
    manifest = prepare()
    print(
        "patients",
        manifest["patient_counts"],
        "annotations",
        manifest["annotation_counts"],
        "skipped",
        manifest["skipped_null_criterion_text"],
    )
