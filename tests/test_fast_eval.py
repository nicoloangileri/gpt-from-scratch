"""The shared prefix is the plain forward computed in another order, nothing more.

A small random network with the real vocabulary and context length stands in for a
checkpoint: what is tested is that the two code paths agree, which does not depend on
the weights being trained. The head is scaled up so that the values of different
moves are far apart compared with float rounding, and an argmax means something.
"""
import chess
import numpy as np
import pytest
import torch

from src.chess_data import ACTION_OFFSET
from src.fast_eval import (move_tokens, network_values, quantize_int8, sample_positions,
                           shared_prefix_logits, state_tokens)
from src.model import GPT, GPTConfig
from src.puzzles_eval import best_move


def tiny_model(seed: int = 0, causal: bool = True) -> GPT:
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=1999, block_size=78, n_layer=2, n_head=2, n_embd=32,
                          causal=causal, num_classes=16)).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(40.0)
    return model


CENTRES = torch.tensor((np.arange(16) + 0.5) / 16, dtype=torch.float32)


def test_one_position_matches_the_plain_forward():
    model, board = tiny_model(), sample_positions(1, seed=3)[0]
    moves = [m.uci() for m in board.legal_moves]
    state, actions = state_tokens(board), move_tokens(moves)
    plain, _ = model(torch.cat([state.repeat(len(moves), 1), actions[:, None]], dim=1))
    fast = shared_prefix_logits(model, state[None], actions)
    assert torch.allclose(fast, plain, atol=1e-5)


def test_many_positions_each_move_reads_its_own_prefix():
    """The owner index is what keeps one position's moves off another's cache: give
    every move the wrong position and the logits must change."""
    model, boards = tiny_model(1), sample_positions(5, seed=4)
    moves = [[m.uci() for m in b.legal_moves] for b in boards]
    fast = network_values(model, CENTRES, boards, moves, "cpu", shared=True)
    plain = network_values(model, CENTRES, boards, moves, "cpu", shared=False)
    for f, p in zip(fast, plain):
        assert f.shape == p.shape and np.allclose(f, p, atol=1e-6)
    states = torch.stack([state_tokens(b) for b in boards])
    actions = torch.cat([move_tokens(m) for m in moves])
    owner = torch.repeat_interleave(torch.arange(5), torch.tensor([len(m) for m in moves]))
    right = shared_prefix_logits(model, states, actions, owner)
    wrong = shared_prefix_logits(model, states, actions, (owner + 1) % 5)
    assert not torch.allclose(right, wrong, atol=1e-3)


def test_the_same_moves_are_chosen():
    model = tiny_model(2)
    for board in sample_positions(12, seed=5):
        assert (best_move(model, CENTRES, board, "cpu", shared=True)
                == best_move(model, CENTRES, board, "cpu", shared=False))


def test_it_refuses_a_model_whose_prefix_sees_the_move():
    """Under full attention the position's hidden states depend on the move, and
    sharing them would be silently wrong."""
    model, board = tiny_model(causal=False), chess.Board()
    with pytest.raises(ValueError):
        shared_prefix_logits(model, state_tokens(board)[None], torch.tensor([ACTION_OFFSET]))


def test_int8_is_close_but_not_the_same_arithmetic():
    model, boards = tiny_model(3), sample_positions(4, seed=6)
    moves = [[m.uci() for m in b.legal_moves] for b in boards]
    q8 = quantize_int8(model)
    assert type(q8.lm_head) is not torch.nn.Linear              # really swapped
    assert type(model.lm_head) is torch.nn.Linear               # and the original untouched
    a = np.concatenate(network_values(model, CENTRES, boards, moves, "cpu"))
    b = np.concatenate(network_values(q8, CENTRES, boards, moves, "cpu"))
    assert np.abs(a - b).max() < 0.1
