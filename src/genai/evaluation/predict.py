"""Run greedy decoding and score criterion predictions. Test data is not read here."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import torch

from genai.constants import ALL_LABELS, EXCLUSION_LABELS, INCLUSION_LABELS
from genai.inference.kv_cache import greedy_decode
from genai.metrics.classification import classification_summary
from genai.metrics.evidence import evidence_summary
from genai.parsing import OutputParseError, parse_prediction
from genai.prompts import build_messages


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _sentences(record: dict) -> dict[int, str]:
    return {int(key): value for key, value in record["sentences"].items()}


def _json_stop(tokenizer):
    def stop(new_ids: torch.Tensor) -> bool:
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        start = text.find("{")
        return start >= 0 and "}" in text[start:]

    return stop


@torch.no_grad()
def predict_record(model, tokenizer, record: dict, max_new_tokens: int, use_cache: bool = True) -> dict:
    messages = build_messages(
        str(record["note"]),
        str(record["criterion_type"]),
        str(record["criterion_text"]),
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(_device(model))
    decoded = greedy_decode(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        eos_id=tokenizer.eos_token_id,
        stop_fn=_json_stop(tokenizer),
    )
    new_ids = decoded.sequences[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    sentences = _sentences(record)
    pred_label = None
    pred_ids: list[int] = []
    error = None
    try:
        parsed = parse_prediction(text, str(record["criterion_type"]))
        pred_label = str(parsed["eligibility"])
        pred_ids = [int(item) for item in parsed["sentence_ids"]]
    except OutputParseError as exc:
        error = str(exc)
    return {
        "annotation_id": record.get("annotation_id"),
        "patient_id": record.get("patient_id"),
        "trial_id": record.get("trial_id"),
        "criterion_type": record["criterion_type"],
        "criterion_text": record["criterion_text"],
        "gold_label": record["expert_eligibility"],
        "gold_sentences": list(record["expert_sentences"]),
        "pred_label": pred_label,
        "pred_sentences": pred_ids,
        "parse_error": error,
        "raw_output": text,
        "note": record["note"],
        "sentences": {str(key): value for key, value in sentences.items()},
        "invalid_sentence_ids": [sentence_id for sentence_id in pred_ids if sentence_id not in sentences],
    }


def predict_records(
    model,
    tokenizer,
    records: Iterable[dict],
    max_new_tokens: int,
    use_cache: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    rows = list(records)
    predictions = []
    total = len(rows)
    for index, record in enumerate(rows, start=1):
        predictions.append(
            predict_record(model, tokenizer, record, max_new_tokens=max_new_tokens, use_cache=use_cache)
        )
        if progress is not None:
            progress(index, total)
    return predictions


def _labels_for_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    gold = [str(row["gold_label"]) for row in rows]
    pred = [str(row["pred_label"]) if row["pred_label"] is not None else "<unparsed>" for row in rows]
    return gold, pred


def summarize_predictions(rows: list[dict]) -> dict[str, object]:
    overall = classification_summary(*_labels_for_rows(rows), list(ALL_LABELS) + ["<unparsed>"])
    inclusion_rows = [row for row in rows if row["criterion_type"] == "inclusion"]
    exclusion_rows = [row for row in rows if row["criterion_type"] == "exclusion"]
    inclusion = classification_summary(
        *_labels_for_rows(inclusion_rows), list(INCLUSION_LABELS) + ["<unparsed>"]
    )
    exclusion = classification_summary(
        *_labels_for_rows(exclusion_rows), list(EXCLUSION_LABELS) + ["<unparsed>"]
    )

    def pairs(subset: list[dict]) -> list[tuple[set[int], set[int]]]:
        return [(set(int(item) for item in row["gold_sentences"]), set(int(item) for item in row["pred_sentences"])) for row in subset]

    return {
        "n": len(rows),
        "overall": overall,
        "inclusion": inclusion,
        "exclusion": exclusion,
        "evidence": {
            "overall": evidence_summary(pairs(rows)),
            "inclusion": evidence_summary(pairs(inclusion_rows)),
            "exclusion": evidence_summary(pairs(exclusion_rows)),
        },
        "unparsed": sum(row["pred_label"] is None for row in rows),
        "invalid_sentence_id_rows": sum(bool(row["invalid_sentence_ids"]) for row in rows),
        "decoding": {"method": "greedy", "use_cache": True, "temperature": 0},
    }


def score_records(
    model,
    tokenizer,
    records: list[dict],
    max_new_tokens: int,
    use_cache: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    rows = predict_records(
        model, tokenizer, records, max_new_tokens=max_new_tokens, use_cache=use_cache, progress=progress
    )
    summary = summarize_predictions(rows)
    summary["predictions"] = rows
    return summary


def write_metrics(path, summary: dict) -> None:
    public = {key: value for key, value in summary.items() if key != "predictions"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public, indent=2), encoding="utf-8")
