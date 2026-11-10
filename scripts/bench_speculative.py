"""Greedy speculative decoding against ordinary target greedy decoding."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import time

import torch

from genai.inference.cache_utils import synchronize
from genai.inference.kv_cache import greedy_decode
from genai.inference.speculative import speculative_decode
from genai.models.load import (
    adapter_ready,
    draft_dir,
    load_causal_lm,
    load_tokenizer,
    target_dir,
)
from genai.paths import ADAPTER_DIR, RESULTS_DIR

PROMPTS = [
    "Patient note:\n0. A 58-year-old woman has chest pain.\n\nReturn JSON only.",
    "Summarize why a missing lab value is not proof of eligibility.",
    "List two reasons an exclusion criterion can be not applicable.",
    "Criterion type: inclusion\nCriterion: Age 18 to 65 years",
]


def main() -> None:
    target_tokenizer = load_tokenizer(target_dir())
    draft_tokenizer = load_tokenizer(draft_dir())
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise SystemExit("Draft and target tokenizers do not share token ids.")
    adapter = ADAPTER_DIR if adapter_ready() else None
    target = load_causal_lm(target_dir(), adapter)
    draft = load_causal_lm(draft_dir(), None)
    device = next(target.parameters()).device
    rows = []
    for text in PROMPTS:
        input_ids = target_tokenizer(text, return_tensors="pt").input_ids.to(device)
        synchronize(device)
        started = time.perf_counter()
        greedy = greedy_decode(target, input_ids, max_new_tokens=32, use_cache=True, eos_id=target_tokenizer.eos_token_id)
        synchronize(device)
        greedy_seconds = time.perf_counter() - started
        speculated = speculative_decode(
            target,
            draft,
            input_ids,
            max_new_tokens=32,
            gamma=4,
            eos_id=target_tokenizer.eos_token_id,
        )
        equal = torch.equal(greedy.sequences, speculated.sequences)
        rows.append(
            {
                "prompt": text,
                "token_equal": equal,
                "greedy_seconds": greedy_seconds,
                "greedy_tokens_per_second": greedy.new_tokens / greedy_seconds if greedy_seconds else 0,
                "speculative_seconds": speculated.seconds,
                "speculative_tokens_per_second": speculated.tokens_per_second,
                "speedup": greedy_seconds / speculated.seconds if speculated.seconds else 0,
                "acceptance_rate": speculated.acceptance_rate,
                "draft_proposed": speculated.draft_proposed,
                "draft_accepted": speculated.draft_accepted,
                "bonus_tokens": speculated.bonus_tokens,
                "peak_memory_bytes": speculated.peak_memory_bytes,
                "new_tokens": speculated.new_tokens,
            }
        )
        print(text[:40], "equal", equal, "accept", round(speculated.acceptance_rate, 3), flush=True)
    payload = {
        "target": "adapter" if adapter else "base",
        "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        "gamma": 4,
        "max_new_tokens": 32,
        "prompts": rows,
        "note": "A speedup is not required. Token equality and the measured rate are the result.",
    }
    path = RESULTS_DIR / "task5" / "speculative.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
