"""Fast CPU tests: shapes, loss, causality, generation. `pytest -q`."""

import torch

from src.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_forward_shapes_and_loss():
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    logits, loss = m(x, x)
    assert logits.shape == (4, CFG.block_size, CFG.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_inference_returns_last_position_only():
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, CFG.block_size))
    logits, loss = m(x)
    assert logits.shape == (2, 1, CFG.vocab_size) and loss is None


def test_causality_future_tokens_do_not_change_past_logits():
    """Changing the last token must not change logits at earlier positions."""
    m = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    with torch.no_grad():
        a, _ = m(x, x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
        b, _ = m(x2, x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_generate_extends_sequence():
    m = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 4))
    out = m.generate(x, max_new_tokens=8, top_k=10)
    assert out.shape == (1, 12)
