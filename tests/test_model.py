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


def test_the_chess_prediction_depends_on_the_move_and_the_position():
    """The action-value head must read the LAST position, where the move token sits.

    Read any earlier position and, with causal attention, the model cannot see the
    move at all: every move of a position gets the same value and move choice becomes
    a coin flip. The suite stayed green with exactly that mutation until this test.
    """
    from src.chess_data import ACTION_OFFSET, FULL_VOCAB, SAMPLE_LENGTH
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH, num_classes=64,
                      n_layer=2, n_head=2, n_embd=32)).eval()
    position = torch.randint(0, ACTION_OFFSET, (1, SAMPLE_LENGTH - 1))
    move_a = torch.cat([position, torch.tensor([[ACTION_OFFSET + 100]])], dim=1)
    move_b = torch.cat([position, torch.tensor([[ACTION_OFFSET + 200]])], dim=1)
    other_position = move_a.clone()
    other_position[0, 5] = (other_position[0, 5] + 1) % ACTION_OFFSET
    with torch.no_grad():
        a, _ = m(move_a)
        b, _ = m(move_b)
        c, _ = m(other_position)
    assert a.shape == (1, 64)
    assert not torch.allclose(a, b)          # same position, different move
    assert not torch.allclose(a, c)          # same move, different position
