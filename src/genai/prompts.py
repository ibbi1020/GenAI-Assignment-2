"""Prompt construction.

The prompt receives the numbered note, the criterion type, the criterion text,
and the label set allowed for that type. GPT-4 fields are rejected if a caller
tries to pass them in.
"""

from __future__ import annotations

from collections.abc import Mapping

from genai.constants import EXCLUSION_LABELS, FORBIDDEN_FIELDS, INCLUSION_LABELS

SYSTEM_PROMPT = (
    "You screen one clinical-trial criterion against a patient note for an educational "
    "prototype. You are not making an enrollment decision. Answer with one JSON object "
    "and no other text. The object has two keys: eligibility (string) and sentence_ids "
    "(list of integers). Choose eligibility only from the allowed labels. sentence_ids "
    "must be ids printed in the note. If the note does not support a citation, use an "
    "empty list. An empty list is not proof that the patient is eligible or ineligible."
)


def labels_for(criterion_type: str) -> tuple[str, ...]:
    kind = criterion_type.strip().lower()
    if kind == "inclusion":
        return INCLUSION_LABELS
    if kind == "exclusion":
        return EXCLUSION_LABELS
    raise ValueError(
        f"Criterion type must be 'inclusion' or 'exclusion'. Received {criterion_type!r}."
    )


def _reject_forbidden(names: object) -> None:
    leaked = sorted(set(names).intersection(FORBIDDEN_FIELDS))
    if leaked:
        raise ValueError(
            "Refusing to build a prompt that includes model-generated dataset fields: "
            + ", ".join(leaked)
        )


def build_messages(note: str, criterion_type: str, criterion_text: str, **extra: object) -> list[dict[str, str]]:
    _reject_forbidden(extra)
    if not note.strip():
        raise ValueError("The patient note is empty.")
    if not criterion_text.strip():
        raise ValueError("The criterion text is empty.")
    allowed = labels_for(criterion_type)
    user = (
        f"Patient note:\n{note.strip()}\n\n"
        f"Criterion type: {criterion_type.strip().lower()}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed eligibility labels: {', '.join(allowed)}\n\n"
        'Return JSON only: {"eligibility": "<label>", "sentence_ids": [<int>, ...]}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def prompt_from_record(record: Mapping[str, object]) -> list[dict[str, str]]:
    """Build messages from a prepared record. Forbidden keys are an error."""
    _reject_forbidden(record.keys())
    return build_messages(
        note=str(record["note"]),
        criterion_type=str(record["criterion_type"]),
        criterion_text=str(record["criterion_text"]),
    )
