"""Training examples. Each real teaching note is used once."""

from __future__ import annotations

import json
from collections import Counter

from genai.prompts import build_messages, labels_for


def answer_weights(records: list[dict]) -> dict[tuple[str, str], float]:
    """Equal total weight for every real (criterion type, answer) pair.

    Each note stays in the teaching set once. A pair with fewer notes gets a
    larger weight on each note, and the weights average to 1. Validation and
    test records must not be passed here.
    """
    counts: Counter[tuple[str, str]] = Counter(
        (str(record["criterion_type"]), str(record["expert_eligibility"])) for record in records
    )
    if not counts:
        return {}
    total = sum(counts.values())
    groups = len(counts)
    return {key: total / (groups * count) for key, count in counts.items()}


def label_counts(records: list[dict]) -> dict[str, int]:
    return dict(Counter(f"{row['criterion_type']}|{row['expert_eligibility']}" for row in records))


def target_json(record: dict) -> str:
    payload = {
        "eligibility": record["expert_eligibility"],
        "sentence_ids": list(record["expert_sentences"]),
    }
    return json.dumps(payload, ensure_ascii=False)


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def fit_note(tokenizer, record: dict, max_prompt_tokens: int) -> tuple[str, dict[int, str], list[int]]:
    """Keep a prefix of sentences so the chat prompt fits. Returns dropped gold ids."""
    sentences = {int(key): value for key, value in record["sentences"].items()}
    ordered = [sentences[key] for key in sorted(sentences)]
    if not ordered:
        messages = build_messages("0. (no note text)", record["criterion_type"], record["criterion_text"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "0. (no note text)", {}, list(record["expert_sentences"])

    best = 0
    best_note = ""
    lo = 0
    hi = len(ordered)
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            note = "0. (note truncated)"
            kept = {0: "(note truncated)"}
        else:
            kept = {index: ordered[index] for index in range(mid)}
            note = "\n".join(f"{index}. {kept[index]}" for index in range(mid))
        messages = build_messages(note, str(record["criterion_type"]), str(record["criterion_text"]))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if _token_count(tokenizer, prompt) <= max_prompt_tokens:
            best = mid
            best_note = note
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        kept_map = {0: "(note truncated)"}
        best_note = "0. (note truncated)"
    else:
        kept_map = {index: ordered[index] for index in range(best)}
    dropped = [sentence_id for sentence_id in record["expert_sentences"] if sentence_id not in kept_map]
    return best_note, kept_map, dropped


def encode_example(tokenizer, record: dict, max_length: int, answer_budget: int = 64) -> dict[str, list[int]] | None:
    note, _kept, _dropped = fit_note(tokenizer, record, max_length - answer_budget)
    messages = build_messages(note, str(record["criterion_type"]), str(record["criterion_text"]))
    messages.append({"role": "assistant", "content": target_json(record)})
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise RuntimeError("Chat template did not keep the prompt as a prefix of the full example.")
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Tokenized prompt is not a prefix of the tokenized example.")
    if len(full_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels, "allowed": list(labels_for(str(record["criterion_type"])))}
