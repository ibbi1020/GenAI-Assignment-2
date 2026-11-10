"""Patient-grouped split required by the assignment.

Sort unique patient ids, shuffle with numpy.random.default_rng(42), then take
the first 37 as train, the next 8 as validation, and the rest as test.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from genai.constants import SPLIT_SEED, TRAIN_PATIENTS, VAL_PATIENTS


def assign_splits(patient_ids: Iterable[str]) -> dict[str, list[str]]:
    unique = sorted(set(patient_ids))
    needed = TRAIN_PATIENTS + VAL_PATIENTS
    if len(unique) < needed:
        raise ValueError(f"Need at least {needed} patients, found {len(unique)}.")
    ids = np.array(unique)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ids)
    train_end = TRAIN_PATIENTS
    val_end = TRAIN_PATIENTS + VAL_PATIENTS
    return {
        "train": ids[:train_end].tolist(),
        "validation": ids[train_end:val_end].tolist(),
        "test": ids[val_end:].tolist(),
    }


def assert_no_patient_leakage(splits: dict[str, list[str]]) -> None:
    names = ("train", "validation", "test")
    sets = {name: set(splits[name]) for name in names}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("A patient appears in more than one split.")
    if len(sets["train"]) != len(splits["train"]):
        raise ValueError("Duplicate patient id inside train.")
    if len(sets["validation"]) != len(splits["validation"]):
        raise ValueError("Duplicate patient id inside validation.")
    if len(sets["test"]) != len(splits["test"]):
        raise ValueError("Duplicate patient id inside test.")
