"""Quote only the sentences the model cited."""

from __future__ import annotations


def evidence_rows(sentence_ids: list[int], sentences: dict[int, str]) -> list[dict[str, object]]:
    return [{"id": sentence_id, "text": sentences[sentence_id]} for sentence_id in sentence_ids]


def evidence_explanation(label: str, sentence_ids: list[int], sentences: dict[int, str]) -> str:
    if not sentence_ids:
        return (
            f"No sentence was cited. The label is '{label}'. "
            "An empty citation is not proof of eligibility or ineligibility."
        )
    quoted = " ".join(f"[{sentence_id}] {sentences[sentence_id]}" for sentence_id in sentence_ids)
    return (
        f"The cited sentences are: {quoted} "
        f"They are the only text used to support the label '{label}'. "
        "Sentences that were not cited were not used."
    )


def missing_information(label: str, sentence_ids: list[int]) -> bool:
    return label == "not enough information" or len(sentence_ids) == 0
