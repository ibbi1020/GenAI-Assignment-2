import numpy as np

from genai.constants import FORBIDDEN_FIELDS
from genai.data.records import build_record
from genai.data.sentences import ensure_numbered, parse_numbered_note
from genai.data.split import assert_no_patient_leakage, assign_splits


def test_split_is_deterministic_and_covers_every_patient():
    patients = [f"p{index:02d}" for index in range(53)]
    first = assign_splits(patients)
    second = assign_splits(patients)
    assert first == second
    assert len(first["train"]) == 37
    assert len(first["validation"]) == 8
    assert len(first["test"]) == 8
    assert_no_patient_leakage(first)
    covered = set(first["train"]) | set(first["validation"]) | set(first["test"])
    assert covered == set(patients)
    assert first["train"] != sorted(patients)[:37]


def test_split_uses_the_required_generator():
    patients = [f"p{index:02d}" for index in range(53)]
    ids = np.array(sorted(patients))
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    assert assign_splits(patients)["train"] == ids[:37].tolist()


def test_empty_numbered_sentence_keeps_its_id():
    parsed = parse_numbered_note("0. Hello. 1. 2. After the gap.")
    assert parsed is not None
    assert parsed[0] == "Hello."
    assert parsed[1] == ""
    assert parsed[2] == "After the gap."
    note = "0. First sentence here. 1. Second sentence here."
    parsed = parse_numbered_note(note)
    assert parsed == {0: "First sentence here.", 1: "Second sentence here."}
    numbered, sentences = ensure_numbered(note)
    assert sentences[1] == "Second sentence here."
    assert numbered.startswith("0. First")


def test_unnumbered_note_gets_ids_from_zero():
    numbered, sentences = ensure_numbered("The patient is 58. She has chest pain.")
    assert sentences[0] == "The patient is 58."
    assert 1 in sentences
    assert numbered.startswith("0. ")


def test_record_drops_gpt4_fields_and_keeps_expert_labels():
    row = {
        "annotation_id": 3,
        "patient_id": "sigir-1",
        "note": "0. A 58-year-old woman. 1. She has pain.",
        "trial_id": "NCT1",
        "trial_title": "A study",
        "criterion_type": "inclusion",
        "criterion_text": "Age 25 to 65 years",
        "expert_eligibility": "included",
        "expert_sentences": "[0]",
        "gpt4_explanation": "secret",
        "explanation_correctness": "Correct",
        "gpt4_sentences": "[0]",
        "gpt4_eligibility": "included",
        "training": True,
    }
    record = build_record(row)
    assert record is not None
    for field in FORBIDDEN_FIELDS:
        assert field not in record
    assert "training" not in record
    assert record["expert_sentences"] == [0]
    assert record["expert_eligibility"] == "included"


def test_saved_manifest_has_no_patient_leakage():
    import json

    from genai.constants import FORBIDDEN_FIELDS
    from genai.data.records import read_jsonl
    from genai.paths import MANIFEST_PATH, PROCESSED_DIR

    if not MANIFEST_PATH.exists():
        return
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["patient_counts"] == {"train": 37, "validation": 8, "test": 8}
    assert manifest["numpy_version"] == "2.2.6"
    assert_no_patient_leakage(manifest["patients"])
    seen = []
    for name in ("train", "validation", "test"):
        rows = read_jsonl(PROCESSED_DIR / f"{name}.jsonl")
        assert len(rows) == manifest["annotation_counts"][name]
        for row in rows:
            assert set(row).isdisjoint(FORBIDDEN_FIELDS)
            assert row["patient_id"] in manifest["patients"][name]
            seen.append(row["patient_id"])
    assert set(seen) == set().union(*(set(ids) for ids in manifest["patients"].values()))


def test_null_criterion_text_is_skipped():
    row = {
        "annotation_id": 1,
        "patient_id": "p",
        "note": "0. Hello.",
        "trial_id": "t",
        "trial_title": "",
        "criterion_type": "inclusion",
        "criterion_text": None,
        "expert_eligibility": "included",
        "expert_sentences": "[]",
    }
    assert build_record(row) is None
