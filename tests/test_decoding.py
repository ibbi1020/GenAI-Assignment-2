from pathlib import Path

import torch

from genai.inference.kv_cache import greedy_decode
from genai.inference.speculative import speculative_decode
from genai.models.tiny import make_tiny

ROOT = Path(__file__).resolve().parents[1]


def test_decode_sources_do_not_call_generate():
    for relative in (
        "src/genai/inference/kv_cache.py",
        "src/genai/inference/speculative.py",
    ):
        source = (ROOT / relative).read_text()
        assert ".generate(" not in source
        assert "assisted_generation" not in source


def test_cached_greedy_matches_uncached_at_three_lengths():
    model = make_tiny(0)
    lengths = ((4, 3), (8, 5), (16, 7))
    for prompt_len, new_tokens in lengths:
        torch.manual_seed(prompt_len)
        prompt = torch.randint(0, model.vocab_size, (1, prompt_len))
        cached = greedy_decode(model, prompt, new_tokens, use_cache=True)
        plain = greedy_decode(model, prompt, new_tokens, use_cache=False)
        assert torch.equal(cached.sequences, plain.sequences)
        assert cached.new_tokens == new_tokens


def test_speculative_tokens_match_target_greedy():
    target = make_tiny(1)
    draft = make_tiny(2)
    torch.manual_seed(9)
    prompt = torch.randint(0, target.vocab_size, (1, 6))
    for gamma in (1, 4):
        greedy = greedy_decode(target, prompt, max_new_tokens=12, use_cache=True)
        speculated = speculative_decode(
            target, draft, prompt, max_new_tokens=12, gamma=gamma
        )
        assert torch.equal(speculated.sequences, greedy.sequences)
        assert speculated.draft_proposed > 0
        assert 0.0 <= speculated.acceptance_rate <= 1.0


def test_speculative_stops_on_eos_like_greedy():
    target = make_tiny(3)
    draft = make_tiny(4)
    prompt = torch.randint(0, target.vocab_size, (1, 3))
    eos_id = 0
    greedy = greedy_decode(target, prompt, max_new_tokens=10, use_cache=True, eos_id=eos_id)
    speculated = speculative_decode(
        target, draft, prompt, max_new_tokens=10, gamma=3, eos_id=eos_id
    )
    assert torch.equal(speculated.sequences, greedy.sequences)


def test_crop_cache_uses_a_negative_token_count():
    from genai.inference.cache_utils import cache_length, crop_cache

    class _Layer:
        def __init__(self):
            self.keys = torch.zeros(1, 1, 5, 2)

        def get_seq_length(self) -> int:
            return int(self.keys.shape[-2])

        def crop(self, tokens_to_remove: int) -> None:
            if tokens_to_remove > 0:
                raise ValueError("positive crop is not an absolute length")
            if tokens_to_remove < 0:
                self.keys = self.keys[..., :tokens_to_remove, :]

    class _Cache:
        def __init__(self):
            self.layer = _Layer()

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self.layer.get_seq_length()

        def crop(self, tokens_to_remove: int) -> None:
            self.layer.crop(tokens_to_remove)

    cache = _Cache()
    cropped = crop_cache(cache, 3)
    assert cache_length(cropped) == 3
    assert crop_cache(cache, 3) is cache
