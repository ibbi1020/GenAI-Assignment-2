"""Screen one criterion. Explanations quote only cited sentences."""

from __future__ import annotations

import torch

from genai.data.sentences import ensure_numbered
from genai.evaluation.predict import _device, _json_stop
from genai.explain import evidence_explanation, evidence_rows, missing_information
from genai.inference.kv_cache import greedy_decode
from genai.parsing import check_sentence_ids, parse_prediction
from genai.prompts import build_messages, labels_for


@torch.no_grad()
def screen_criterion(model, tokenizer, note: str, criterion_type: str, criterion_text: str, max_new_tokens: int = 128) -> dict:
    if not note or not note.strip():
        raise ValueError("The patient note is empty. Paste a note, or load an example.")
    if not criterion_text or not criterion_text.strip():
        raise ValueError("The criterion text is empty.")
    labels_for(criterion_type)
    numbered, sentences = ensure_numbered(note)
    if not sentences:
        raise ValueError("The patient note did not contain any sentences.")
    messages = build_messages(numbered, criterion_type, criterion_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(_device(model))
    decoded = greedy_decode(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_id=tokenizer.eos_token_id,
        stop_fn=_json_stop(tokenizer),
    )
    text = tokenizer.decode(decoded.sequences[0, input_ids.shape[1] :], skip_special_tokens=True)
    parsed = parse_prediction(text, criterion_type)
    sentence_ids = list(parsed["sentence_ids"])
    check_sentence_ids(sentence_ids, sentences)
    label = str(parsed["eligibility"])
    return {
        "ok": True,
        "numbered_note": numbered,
        "criterion_type": criterion_type.strip().lower(),
        "criterion_text": criterion_text.strip(),
        "eligibility": label,
        "sentence_ids": sentence_ids,
        "evidence": evidence_rows(sentence_ids, sentences),
        "explanation": evidence_explanation(label, sentence_ids, sentences),
        "missing_information": missing_information(label, sentence_ids),
        "raw_output": text,
    }
