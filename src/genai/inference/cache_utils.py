"""Cache length and cropping for tuple caches and Hugging Face cache objects."""

from __future__ import annotations

import torch


def cache_length(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    key = past[0][0]
    return int(key.shape[-2])


def crop_cache(past, length: int):
    """Keep the first `length` cached positions. Returns the cache to keep using.

    Current Hugging Face caches treat ``crop`` as a negative count of tokens to
    drop. Older tuple caches are sliced to an absolute length instead.
    """
    if past is None:
        return None
    if hasattr(past, "crop") and hasattr(past, "get_seq_length"):
        current = cache_length(past)
        remove = current - int(length)
        if remove < 0:
            raise ValueError(f"Cannot crop a cache of length {current} to {length}.")
        if remove > 0:
            past.crop(-remove)
        return past
    cropped = []
    for key, value in past:
        cropped.append((key[:, :, :length, :], value[:, :, :length, :]))
    return tuple(cropped)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def allocated_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        current = int(torch.mps.current_allocated_memory())
        driver = int(torch.mps.driver_allocated_memory()) if hasattr(torch.mps, "driver_allocated_memory") else 0
        return max(current, driver)
    return 0


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def position_ids(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).view(1, -1)


def model_step(model, input_ids: torch.Tensor, past, positions: torch.Tensor | None):
    past_len = 0 if past is None else cache_length(past)
    attention_mask = torch.ones(
        input_ids.shape[0],
        past_len + input_ids.shape[-1],
        dtype=torch.long,
        device=input_ids.device,
    )
    kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "attention_mask": attention_mask,
    }
    if past is not None:
        kwargs["past_key_values"] = past
    if positions is not None:
        kwargs["position_ids"] = positions
    return model(**kwargs)
