"""KV-cache benchmark on the real target model. Three prompt/output lengths."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gc
import json

import torch

from genai.constants import KV_LENGTH_PAIRS
from genai.inference.cache_utils import synchronize
from genai.inference.kv_cache import greedy_decode
from genai.models.load import adapter_ready, load_causal_lm, load_tokenizer, target_dir
from genai.paths import ADAPTER_DIR, RESULTS_DIR


def _time_pair(model, prompt: torch.Tensor, new_tokens: int, repeats: int) -> dict:
    device = prompt.device

    def run(use_cache: bool):
        synchronize(device)
        result = greedy_decode(model, prompt, new_tokens, use_cache=use_cache, eos_id=None)
        synchronize(device)
        return result

    for _ in range(1):
        run(True)
        run(False)
    cached_runs = [run(True) for _ in range(repeats)]
    plain_runs = [run(False) for _ in range(repeats)]
    equal = all(torch.equal(cached.sequences, plain.sequences) for cached, plain in zip(cached_runs, plain_runs, strict=True))
    mismatch = None
    if not equal:
        left = cached_runs[0].sequences[0].tolist()
        right = plain_runs[0].sequences[0].tolist()
        index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        mismatch = {"index": index, "cached_token": None if index is None else left[index], "uncached_token": None if index is None else right[index]}
    def pack(runs):
        return {
            "prefill_seconds": sum(item.prefill_seconds for item in runs) / len(runs),
            "decode_seconds": sum(item.decode_seconds for item in runs) / len(runs),
            "tokens_per_second": sum(item.tokens_per_second for item in runs) / len(runs),
            "peak_memory_bytes": max(item.peak_memory_bytes for item in runs),
            "new_tokens": runs[-1].new_tokens,
        }
    return {"token_equal": equal, "mismatch": mismatch, "cached": pack(cached_runs), "uncached": pack(plain_runs)}


def main() -> None:
    tokenizer = load_tokenizer(target_dir())
    adapter = ADAPTER_DIR if adapter_ready() else None
    model = load_causal_lm(target_dir(), adapter)
    device = next(model.parameters()).device
    rows = []
    for prompt_len, new_tokens in KV_LENGTH_PAIRS:
        torch.manual_seed(prompt_len)
        prompt = torch.randint(0, tokenizer.vocab_size, (1, prompt_len), device=device)
        print(f"kv {prompt_len}/{new_tokens}", flush=True)
        repeats = 2 if prompt_len >= 512 else 3
        rows.append({"prompt_tokens": prompt_len, "new_tokens": new_tokens, **_time_pair(model, prompt, new_tokens, repeats)})
    payload = {
        "model": "adapter" if adapter else "base",
        "eos_disabled": True,
        "reason": "EOS is disabled so both paths emit the requested number of tokens.",
        "warmup": 1,
        "device": device.type,
        "pairs": rows,
    }
    path = RESULTS_DIR / "task3" / "kv_cache.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
