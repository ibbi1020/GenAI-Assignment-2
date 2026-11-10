"""Attention timing. A pure PyTorch tile is not expected to beat a fused kernel."""

from __future__ import annotations

import time

import torch

from genai.inference.cache_utils import synchronize
from genai.models.attention import reference_attention, tiled_online_softmax_attention


def run_attention_case(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    block_size: int,
    causal: bool,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    query = torch.randn(batch, heads, seq, dim, device=device)
    key = torch.randn(batch, heads, seq, dim, device=device)
    value = torch.randn(batch, heads, seq, dim, device=device)
    reference = reference_attention(query, key, value, causal=causal)
    tiled = tiled_online_softmax_attention(query, key, value, causal=causal, block_size=block_size)
    difference = (reference - tiled).abs()

    def time_call(fn) -> float:
        for _ in range(1):
            fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        synchronize(device)
        return (time.perf_counter() - start) / repeats

    reference_seconds = time_call(lambda: reference_attention(query, key, value, causal=causal))
    tiled_seconds = time_call(
        lambda: tiled_online_softmax_attention(query, key, value, causal=causal, block_size=block_size)
    )
    return {
        "batch": batch,
        "heads": heads,
        "seq": seq,
        "dim": dim,
        "block_size": block_size,
        "causal": causal,
        "repeats": repeats,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "reference_seconds": reference_seconds,
        "tiled_seconds": tiled_seconds,
        "note": (
            "A pure PyTorch tiled loop moves one block at a time in Python. "
            "It is not expected to beat a fused attention kernel. The numbers "
            "are a correctness check and an honest runtime comparison."
        ),
    }
