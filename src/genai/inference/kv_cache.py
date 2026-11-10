"""Greedy decoding that calls the model forward itself.

The cached path stores past key/value tensors, feeds one new token at a time,
and sets position ids to the absolute length. The uncached path resends the
whole prefix with use_cache False. Neither path calls model.generate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from genai.inference.cache_utils import allocated_bytes, model_step, position_ids, reset_memory_stats, synchronize


@dataclass
class DecodeResult:
    sequences: torch.Tensor
    prefill_seconds: float
    decode_seconds: float
    new_tokens: int
    peak_memory_bytes: int

    @property
    def tokens_per_second(self) -> float:
        if self.decode_seconds <= 0:
            return 0.0
        return self.new_tokens / self.decode_seconds


def _token(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _stop(token: torch.Tensor, eos_id: int | None) -> bool:
    return eos_id is not None and int(token[0, 0].item()) == int(eos_id)


@torch.no_grad()
def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool,
    eos_id: int | None = None,
    stop_fn=None,
) -> DecodeResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("greedy_decode expects a batch of one sequence.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    device = input_ids.device
    reset_memory_stats(device)
    if not use_cache:
        return _uncached(model, input_ids, max_new_tokens, eos_id, device, stop_fn)
    return _cached(model, input_ids, max_new_tokens, eos_id, device, stop_fn)


def _finished(token, eos_id, produced: torch.Tensor, stop_fn) -> bool:
    if _stop(token, eos_id):
        return True
    return stop_fn is not None and bool(stop_fn(produced))


def _uncached(model, input_ids, max_new_tokens, eos_id, device, stop_fn) -> DecodeResult:
    sequences = input_ids
    prompt_len = input_ids.shape[1]
    start = _now(device)
    output = model(input_ids=sequences, use_cache=False)
    prefill = _now(device) - start
    next_token = _token(output.logits)
    sequences = torch.cat([sequences, next_token], dim=1)
    new_tokens = 1
    decode_start = _now(device)
    while new_tokens < max_new_tokens and not _finished(next_token, eos_id, sequences[:, prompt_len:], stop_fn):
        output = model(input_ids=sequences, use_cache=False)
        next_token = _token(output.logits)
        sequences = torch.cat([sequences, next_token], dim=1)
        new_tokens += 1
    decode = _now(device) - decode_start if new_tokens > 1 else 0.0
    return DecodeResult(sequences, prefill, decode, new_tokens, allocated_bytes(device))


def _cached(model, input_ids, max_new_tokens, eos_id, device, stop_fn) -> DecodeResult:
    prompt_len = input_ids.shape[1]
    synchronize(device)
    start = _now(device)
    output = model_step(model, input_ids, None, position_ids(0, prompt_len, device))
    synchronize(device)
    prefill = _now(device) - start
    past = output.past_key_values
    next_token = _token(output.logits)
    generated = [next_token]
    cursor = prompt_len
    decode_start = _now(device)
    while len(generated) < max_new_tokens and not _finished(
        next_token, eos_id, torch.cat(generated, dim=1), stop_fn
    ):
        output = model_step(model, next_token, past, position_ids(cursor, 1, device))
        past = output.past_key_values
        cursor += 1
        next_token = _token(output.logits)
        generated.append(next_token)
    synchronize(device)
    decode = _now(device) - decode_start if len(generated) > 1 else 0.0
    sequences = torch.cat([input_ids, *generated], dim=1)
    return DecodeResult(sequences, prefill, decode, len(generated), allocated_bytes(device))


def _now(device: torch.device) -> float:
    synchronize(device)
    import time

    return time.perf_counter()
