"""Download the prescribed dataset and write the fixed patient split."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from datasets import load_dataset

from genai.constants import DATASET_ID, FORBIDDEN_FIELDS, SPLIT_SEED
from genai.data.records import build_records, write_jsonl
from genai.data.split import assert_no_patient_leakage, assign_splits
from genai.paths import MANIFEST_PATH, PROCESSED_DIR


def prepare() -> dict[str, object]:
    dataset = load_dataset(DATASET_ID, split="train")
    rows = [dict(row) for row in dataset]
    records, skipped = build_records(rows)
    patient_ids = [str(record["patient_id"]) for record in records]
    splits = assign_splits(patient_ids)
    assert_no_patient_leakage(splits)
    by_patient: dict[str, list[dict[str, object]]] = {name: [] for name in splits}
    lookup = {name: set(ids) for name, ids in splits.items()}
    for record in records:
        placed = False
        for name, ids in lookup.items():
            if record["patient_id"] in ids:
                by_patient[name].append(record)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Patient {record['patient_id']} was not assigned a split.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    annotation_counts = {}
    for name, split_records in by_patient.items():
        write_jsonl(PROCESSED_DIR / f"{name}.jsonl", split_records)
        annotation_counts[name] = len(split_records)
    manifest = {
        "dataset": DATASET_ID,
        "numpy_version": np.__version__,
        "seed": SPLIT_SEED,
        "generator": "numpy.random.default_rng",
        "method": (
            "Sort unique patient ids, shuffle with numpy.random.default_rng(42), "
            "assign the first 37 to train, the next 8 to validation, and the rest to test."
        ),
        "patients": splits,
        "patient_counts": {name: len(ids) for name, ids in splits.items()},
        "annotation_counts": annotation_counts,
        "skipped_null_criterion_text": skipped,
        "label_counts": dict(Counter(str(record["expert_eligibility"]) for record in records)),
        "forbidden_fields_excluded": list(FORBIDDEN_FIELDS),
        "ignored_dataset_column": "training",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
