"""Turn dataset rows into the records the rest of the code is allowed to see."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from genai.constants import ALL_LABELS, FORBIDDEN_FIELDS
from genai.data.sentences import ensure_numbered

ALLOWED_OUTPUT_KEYS = (
    "annotation_id",
    "patient_id",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
    "sentences",
    "note",
)


def parse_sentence_id_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if text in {"", "[]", "None", "null"}:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Sentence id field is not a list: {text}")
    return [int(item) for item in parsed]


def _missing_text(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "none"


def build_record(row: Mapping[str, object]) -> dict[str, object] | None:
    """Build one public record, or None when the criterion text is missing.

    Raises if a forbidden GPT-4 field would be copied into the record.
    """
    if any(name in row and name in FORBIDDEN_FIELDS for name in FORBIDDEN_FIELDS):
        # The raw dataset contains these columns. We refuse to copy them.
        pass
    leaked = [key for key in row if key in FORBIDDEN_FIELDS and key in ALLOWED_OUTPUT_KEYS]
    if leaked:
        raise ValueError(f"Refusing to store forbidden fields: {leaked}")
    if _missing_text(row.get("criterion_text")):
        return None
    label = str(row["expert_eligibility"]).strip().lower()
    if label not in ALL_LABELS:
        raise ValueError(f"Unexpected expert label: {label}")
    criterion_type = str(row["criterion_type"]).strip().lower()
    if criterion_type not in {"inclusion", "exclusion"}:
        raise ValueError(f"Unexpected criterion type: {criterion_type}")
    note, sentences = ensure_numbered(str(row["note"]))
    return {
        "annotation_id": int(row["annotation_id"]),
        "patient_id": str(row["patient_id"]),
        "trial_id": str(row["trial_id"]),
        "trial_title": str(row.get("trial_title") or ""),
        "criterion_type": criterion_type,
        "criterion_text": str(row["criterion_text"]).strip(),
        "expert_eligibility": label,
        "expert_sentences": parse_sentence_id_list(row.get("expert_sentences")),
        "sentences": {str(sentence_id): text for sentence_id, text in sentences.items()},
        "note": note,
    }


def build_records(rows: Iterable[Mapping[str, object]]) -> tuple[list[dict[str, object]], int]:
    records = []
    skipped = 0
    for row in rows:
        record = build_record(row)
        if record is None:
            skipped += 1
            continue
        records.append(record)
    return records, skipped


def write_jsonl(path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
