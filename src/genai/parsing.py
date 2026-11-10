"""Read the JSON object the model was trained to emit."""

from __future__ import annotations

import json
import re

from genai.prompts import labels_for

_INT = re.compile(r"^-?\d+$")


class OutputParseError(ValueError):
    """The model text is not the required JSON object."""


def parse_prediction(text: str, criterion_type: str) -> dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise OutputParseError("Model output did not contain a JSON object.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"Model output was not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise OutputParseError("Model output JSON was not an object.")
    if "eligibility" not in payload or "sentence_ids" not in payload:
        raise OutputParseError("JSON must contain eligibility and sentence_ids.")
    label = str(payload["eligibility"]).strip().lower()
    allowed = labels_for(criterion_type)
    if label not in allowed:
        raise OutputParseError(
            f"The model returned a label that is not allowed for this criterion type: {label}. "
            f"Allowed labels are: {', '.join(allowed)}."
        )
    raw_ids = payload["sentence_ids"]
    if not isinstance(raw_ids, list):
        raise OutputParseError("sentence_ids must be a list.")
    sentence_ids: list[int] = []
    for item in raw_ids:
        if isinstance(item, bool) or not isinstance(item, int):
            text_item = str(item).strip()
            if not _INT.match(text_item):
                raise OutputParseError(f"Sentence id {item!r} is not an integer.")
            item = int(text_item)
        sentence_ids.append(int(item))
    return {"eligibility": label, "sentence_ids": sentence_ids}


def check_sentence_ids(sentence_ids: list[int], sentences: dict[int, str]) -> None:
    missing = [sentence_id for sentence_id in sentence_ids if sentence_id not in sentences]
    if missing:
        known = ", ".join(str(sentence_id) for sentence_id in sorted(sentences)) or "(none)"
        raise OutputParseError(
            f"Sentence id {missing[0]} is not in the numbered note. Known ids: {known}."
        )
