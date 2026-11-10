"""Greedy speculative decoding.

The draft model proposes a short run of tokens. The target model checks that
run in one forward pass. Matching tokens are kept. On the first mismatch the
target's own token replaces the draft token, and both caches are cropped back
to that point. If the whole draft matches, one extra target token is appended.
For greedy decoding the emitted tokens are the same tokens ordinary target
greedy decoding would emit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from genai.inference.cache_utils import (
    allocated_bytes,
    cache_length,
    crop_cache,
    model_step,
    position_ids,
    reset_memory_stats,
    synchronize,
)


@dataclass
class SpeculativeResult:
    sequences: torch.Tensor
    prompt_length: int
    draft_proposed: int
    draft_accepted: int
    bonus_tokens: int
    seconds: float
    peak_memory_bytes: int

    @property
    def new_tokens(self) -> int:
        return int(self.sequences.shape[1] - self.prompt_length)

    @property
    def acceptance_rate(self) -> float:
        if self.draft_proposed == 0:
            return 0.0
        return self.draft_accepted / self.draft_proposed

    @property
    def tokens_per_second(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.new_tokens / self.seconds


def _as_token(logits_row: torch.Tensor) -> int:
    return int(logits_row.argmax(dim=-1).reshape(-1)[0].item())


@torch.no_grad()
def speculative_decode(
    target,
    draft,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    eos_id: int | None = None,
) -> SpeculativeResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("speculative_decode expects a batch of one sequence.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if gamma < 1:
        raise ValueError("gamma must be at least 1.")
    device = input_ids.device
    prompt_length = int(input_ids.shape[1])
    reset_memory_stats(device)
    synchronize(device)
    import time

    started = time.perf_counter()
    target_out = model_step(target, input_ids, None, position_ids(0, prompt_length, device))
    draft_out = model_step(draft, input_ids, None, position_ids(0, prompt_length, device))
    target_past = target_out.past_key_values
    draft_past = draft_out.past_key_values
    target_logits = target_out.logits[:, -1, :]
    draft_logits = draft_out.logits[:, -1, :]
    generated: list[int] = []
    proposed = 0
    accepted = 0
    bonus_tokens = 0

    while len(generated) < max_new_tokens:
        space = max_new_tokens - len(generated)
        look_ahead = min(gamma, space)
        base = prompt_length + len(generated)
        draft_tokens: list[int] = []
        for index in range(look_ahead):
            token_id = _as_token(draft_logits)
            draft_tokens.append(token_id)
            proposed += 1
            if eos_id is not None and token_id == int(eos_id):
                break
            step = torch.tensor([[token_id]], dtype=input_ids.dtype, device=device)
            draft_out = model_step(draft, step, draft_past, position_ids(base + index, 1, device))
            draft_past = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]

        drafted = torch.tensor([draft_tokens], dtype=input_ids.dtype, device=device)
        verified = model_step(target, drafted, target_past, position_ids(base, len(draft_tokens), device))
        target_past = verified.past_key_values
        matched = 0
        corrected: int | None = None
        for index, token_id in enumerate(draft_tokens):
            if index == 0:
                prediction = _as_token(target_logits)
            else:
                prediction = _as_token(verified.logits[0, index - 1])
            if prediction == token_id:
                matched += 1
            else:
                corrected = prediction
                break

        if corrected is None:
            emit = list(draft_tokens)
            if len(emit) < space:
                extra = _as_token(verified.logits[0, -1])
                emit.append(extra)
                bonus_tokens += 1
        else:
            emit = draft_tokens[:matched] + [corrected]

        final_emit: list[int] = []
        stop = False
        for token_id in emit:
            if len(final_emit) >= space:
                break
            final_emit.append(token_id)
            if eos_id is not None and token_id == int(eos_id):
                stop = True
                break
        kept_draft = 0
        for token_id in final_emit:
            if kept_draft < matched and token_id == draft_tokens[kept_draft]:
                kept_draft += 1
            else:
                break
        accepted += kept_draft

        if corrected is None and not stop and len(final_emit) == len(draft_tokens) + 1:
            bonus_id = final_emit[-1]
            bonus_tensor = torch.tensor([[bonus_id]], dtype=input_ids.dtype, device=device)
            bonus_pos = position_ids(base + len(draft_tokens), 1, device)
            target_out = model_step(target, bonus_tensor, target_past, bonus_pos)
            target_past = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :]
            draft_out = model_step(draft, bonus_tensor, draft_past, bonus_pos)
            draft_past = draft_out.past_key_values
            draft_logits = draft_out.logits[:, -1, :]
        elif corrected is None and not stop and len(final_emit) == len(draft_tokens):
            target_logits = verified.logits[:, -1, :]
        else:
            keep = base + kept_draft
            target_past = crop_cache(target_past, keep)
            draft_past = crop_cache(draft_past, keep)
            if len(final_emit) > kept_draft:
                repaired = torch.tensor([final_emit[kept_draft:]], dtype=input_ids.dtype, device=device)
                repair_pos = position_ids(keep, repaired.shape[1], device)
                target_out = model_step(target, repaired, target_past, repair_pos)
                target_past = target_out.past_key_values
                target_logits = target_out.logits[:, -1, :]
                draft_out = model_step(draft, repaired, draft_past, repair_pos)
                draft_past = draft_out.past_key_values
                draft_logits = draft_out.logits[:, -1, :]

        generated.extend(final_emit)
        if cache_length(target_past) != prompt_length + len(generated):
            raise RuntimeError(
                "Target cache length "
                f"{cache_length(target_past)} does not match the emitted sequence "
                f"{prompt_length + len(generated)}."
            )
        if stop:
            break

    synchronize(device)
    elapsed = time.perf_counter() - started
    sequences = torch.cat(
        [input_ids, torch.tensor([generated], dtype=input_ids.dtype, device=device)],
        dim=1,
    ) if generated else input_ids
    return SpeculativeResult(
        sequences=sequences,
        prompt_length=prompt_length,
        draft_proposed=proposed,
        draft_accepted=accepted,
        bonus_tokens=bonus_tokens,
        seconds=elapsed,
        peak_memory_bytes=allocated_bytes(device),
    )
