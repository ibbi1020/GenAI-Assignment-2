from pathlib import Path

import torch

from genai.models.attention import reference_attention, tiled_online_softmax_attention

ROOT = Path(__file__).resolve().parents[1]


def test_attention_module_does_not_call_a_fused_kernel():
    source = (ROOT / "src" / "genai" / "models" / "attention.py").read_text()
    assert "scaled_dot_product_attention" not in source
    assert "flash_attn" not in source


def test_tiled_attention_matches_reference_across_shapes():
    torch.manual_seed(0)
    shapes = [(1, 1, 1, 4), (2, 4, 17, 8), (1, 2, 8, 16), (1, 1, 5, 4)]
    for batch, heads, seq, dim in shapes:
        query = torch.randn(batch, heads, seq, dim)
        key = torch.randn(batch, heads, seq, dim)
        value = torch.randn(batch, heads, seq, dim)
        for causal in (False, True):
            for block in (1, 3, 32, seq + 5):
                reference = reference_attention(query, key, value, causal=causal)
                tiled = tiled_online_softmax_attention(
                    query, key, value, causal=causal, block_size=block
                )
                assert torch.allclose(reference, tiled, atol=1e-5, rtol=1e-5)


def test_causal_mask_hides_future_keys():
    torch.manual_seed(1)
    query = torch.randn(1, 1, 6, 4)
    key = torch.randn(1, 1, 6, 4)
    value = torch.randn(1, 1, 6, 4)
    original = reference_attention(query, key, value, causal=True)
    changed_key = key.clone()
    changed_key[:, :, -1, :] += 3
    changed = reference_attention(query, changed_key, value, causal=True)
    assert torch.allclose(original[:, :, :-1, :], changed[:, :, :-1, :], atol=1e-6)
    assert not torch.allclose(original[:, :, -1, :], changed[:, :, -1, :])
