"""Q(s, a) for many moves at the cost of one position: the prefix is shared.

    python -m src.fast_eval --ckpt checkpoints/chess_500M.pt --positions 30

An input is 77 position tokens followed by one action token, and attention is
causal. So the hidden states of the 77 position tokens never see the action: they
are the same for every legal move of a position. Evaluating 35 moves the plain way
pushes 35 x 78 = 2,730 token positions through the network; running the position
once and then each move as a single token that attends to the cached keys and values
is 77 + 35 = 112. That is the KV-cache of autoregressive decoding, used here for
breadth (many continuations of one prefix) instead of length.

It is exact, not an approximation: the same weights do the same arithmetic in a
different order, so the logits agree with the plain forward to float rounding. The
tests check that and that the chosen moves are the same.

Several positions can go in one call (the children of a search node): the prefixes
run as one batch and each move reads the keys and values of its own position.

int8 dynamic quantization (`quantize_int8`) is the other lever, on CPU only: the
Linear weights are stored as int8 and the activations quantized on the fly. Unlike
the prefix sharing it changes the numbers, so its cost is measured on puzzles rather
than assumed.
"""
from __future__ import annotations

import argparse
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chess_data import ACTION_OFFSET, MOVE_TO_ACTION, tokenize_fen


def _heads(t: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, C = t.shape
    return t.view(B, T, n_head, C // n_head).transpose(1, 2)


@torch.no_grad()
def shared_prefix_logits(model, states: torch.Tensor, actions: torch.Tensor,
                         owner: torch.Tensor | None = None) -> torch.Tensor:
    """Logits for (states[owner[j]], actions[j]) for every j.

    states: (S, T) position tokens; actions: (N,) action tokens (ACTION_OFFSET
    included); owner: (N,) index of each action's position, all zeros if None.
    Equal to model(cat(states[owner], actions[:, None], dim=1)) up to rounding.
    """
    cfg = model.cfg
    if not cfg.causal or cfg.num_classes is None:
        raise ValueError("the prefix is only shared under causal attention with a "
                         "classification head")
    S, T = states.shape
    if T + 1 > cfg.block_size:
        raise ValueError(f"{T} position tokens + 1 action > block_size {cfg.block_size}")
    if owner is None:
        owner = torch.zeros(len(actions), dtype=torch.long, device=actions.device)
    tr = model.transformer
    pos = torch.arange(T + 1, device=states.device)
    xp = tr.wte(states) + tr.wpe(pos[:T])                 # (S, T, C) the positions
    xa = tr.wte(actions[:, None]) + tr.wpe(pos[T:])       # (N, 1, C) one token per move
    for block in tr.h:
        attn = block.attn
        C, H = attn.n_embd, attn.n_head
        qp, kp, vp = (_heads(t, H) for t in attn.c_attn(block.ln_1(xp)).split(C, dim=2))
        qa, ka, va = (_heads(t, H) for t in attn.c_attn(block.ln_1(xa)).split(C, dim=2))
        yp = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True)
        # The move token is last, so causal attention lets it see everything: its own
        # position's cached keys and values plus itself, and no mask is needed.
        k = torch.cat([kp[owner], ka], dim=2)
        v = torch.cat([vp[owner], va], dim=2)
        ya = F.scaled_dot_product_attention(qa, k, v)
        xp = xp + attn.c_proj(yp.transpose(1, 2).reshape(S, T, C))
        xa = xa + attn.c_proj(ya.transpose(1, 2).reshape(len(actions), 1, C))
        xp = xp + block.mlp(block.ln_2(xp))
        xa = xa + block.mlp(block.ln_2(xa))
    return model.lm_head(tr.ln_f(xa)[:, 0, :])


def move_tokens(moves: list[str]) -> torch.Tensor:
    return torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves], dtype=torch.long)


def state_tokens(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(tokenize_fen(board.fen()).astype(np.int64))


@torch.no_grad()
def network_values(model, centres, boards: list[chess.Board], moves: list[list[str]],
                   device: str, shared: bool = True) -> list[np.ndarray]:
    """Expected win probability of every listed move of every board, one network call.

    Network output only: terminal states are the caller's business, as in
    Searcher._q_values. `shared=False` is the plain forward, for checking.
    """
    sizes = [len(m) for m in moves]
    if sum(sizes) == 0:
        return [np.zeros(0) for _ in boards]
    states = torch.stack([state_tokens(b) for b in boards]).to(device)
    actions = torch.cat([move_tokens(m) for m in moves if m]).to(device)
    owner = torch.repeat_interleave(torch.arange(len(boards)), torch.tensor(sizes)).to(device)
    if shared:
        logits = shared_prefix_logits(model, states, actions, owner)
    else:
        logits, _ = model(torch.cat([states[owner], actions[:, None]], dim=1))
    v = (logits.float().softmax(-1) * centres).sum(-1).cpu().numpy().astype(np.float64)
    return list(np.split(v, np.cumsum(sizes)[:-1]))


def quantize_int8(model) -> nn.Module:
    """A copy of `model` with int8 Linear layers (dynamic quantization, CPU only)."""
    import copy
    import warnings
    engines = torch.backends.quantized.supported_engines
    for engine in ("fbgemm", "x86", "qnnpack"):         # x86 servers, then ARM (Apple)
        if engine in engines:
            torch.backends.quantized.engine = engine
            break
    with warnings.catch_warnings():     # torch.ao is deprecated in favour of torchao
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)


def bench(model, centres, boards: list[chess.Board], device: str, shared: bool,
          repeats: int = 3) -> float:
    """Median milliseconds to value every legal move of one position."""
    moves = [[m.uci() for m in b.legal_moves] for b in boards]
    network_values(model, centres, boards[:1], moves[:1], device, shared)   # warm-up
    times = []
    for _ in range(repeats):
        for b, m in zip(boards, moves):
            t = time.perf_counter()
            network_values(model, centres, [b], [m], device, shared)
            times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def sample_positions(n: int, seed: int = 0) -> list[chess.Board]:
    """Positions from random-but-legal play, a stand-in for a game's middle."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(int(rng.integers(6, 40))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if not board.is_game_over():
            out.append(board)
    return out


if __name__ == "__main__":
    from .puzzles_eval import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_500M.pt")
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, centres, _ = load_model(args.ckpt, "cpu")
    boards = sample_positions(args.positions)
    n_moves = np.mean([b.legal_moves.count() for b in boards])
    q8 = quantize_int8(model)
    rows = {"fp32, plain forward": (model, False), "fp32, shared prefix": (model, True),
            "int8, plain forward": (q8, False), "int8, shared prefix": (q8, True)}
    base = None
    print(f"[bench] {len(boards)} positions, {n_moves:.1f} legal moves on average, "
          f"{torch.get_num_threads()} CPU threads")
    for name, (m, shared) in rows.items():
        ms = bench(m, centres, boards, "cpu", shared)
        base = base or ms
        print(f"  {name:22s} {ms:7.2f} ms per position  ({base / ms:4.1f}x)")
