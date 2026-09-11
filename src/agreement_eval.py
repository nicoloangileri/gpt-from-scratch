"""How often does the model pick the move Stockfish would, and what does it cost?

    python -m src.agreement_eval --ckpt checkpoints/chess_best.pt --positions 500

Puzzle accuracy is all-or-nothing over a whole forced line, which makes it a
coarse instrument. This measures the model move by move on held-out annotated
positions: how often it picks the top move, and — the number that maps to
playing strength — how much win probability it gives up when it does not.

The test shard stores one record per (position, legal move), shuffled. Grouping
the whole file by FEN puts each position back together with every move scored,
which is what makes this comparison possible at all.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import chess
import numpy as np
import torch

from .chess_data import ACTION_OFFSET, MOVE_TO_ACTION, iter_action_value, tokenize_fen
from .puzzles_eval import load_model


@torch.no_grad()
def _score_moves(model, centres, fen: str, moves: list[str], device: str) -> np.ndarray:
    """Expected win probability for each candidate move, in one batch."""
    state = torch.from_numpy(tokenize_fen(fen).astype(np.int64))
    actions = torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves], dtype=torch.long)
    x = torch.cat([state.repeat(len(moves), 1), actions[:, None]], dim=1).to(device)
    logits, _ = model(x)
    return (logits.softmax(-1) * centres).sum(-1).cpu().numpy()


def run(ckpt: str, bag: str, positions: int, device: str, blunder: float = 0.15) -> dict:
    print(f"[agreement] grouping {bag} by position...", flush=True)
    groups: dict[str, list] = defaultdict(list)
    for fen, move, win_prob in iter_action_value(bag):
        groups[fen].append((move, win_prob))
    print(f"[agreement] {len(groups):,} distinct positions", flush=True)

    model, centres, ck = load_model(ckpt, device)
    top1 = top3 = tested = 0
    losses = []
    for fen, records in groups.items():
        if tested >= positions:
            break
        board = chess.Board(fen)
        legal = {m.uci() for m in board.legal_moves}
        moves = [m for m, _ in records]
        if set(moves) != legal or len(moves) < 2:      # only fully annotated positions
            continue
        truth = np.array([p for _, p in records])
        predicted = _score_moves(model, centres, fen, moves, device)

        best_truth = int(truth.argmax())
        chosen = int(predicted.argmax())
        top1 += chosen == best_truth
        top3 += chosen in set(np.argsort(truth)[::-1][:3].tolist())
        losses.append(float(truth[best_truth] - truth[chosen]))
        tested += 1

    losses = np.array(losses)
    out = {
        "positions": tested,
        "top1_pct": 100 * top1 / max(1, tested),
        "top3_pct": 100 * top3 / max(1, tested),
        "mean_loss": float(losses.mean()),
        "median_loss": float(np.median(losses)),
        "blunder_pct": 100 * float((losses > blunder).mean()),
    }
    print(f"[agreement] {out['positions']} positions (checkpoint step {ck.get('step')})")
    print(f"  picks Stockfish's best move : {out['top1_pct']:.1f}%")
    print(f"  picks one of its top 3      : {out['top3_pct']:.1f}%")
    print(f"  win probability given up    : mean {out['mean_loss']:.4f}, "
          f"median {out['median_loss']:.4f}")
    print(f"  moves losing > {blunder:.2f}       : {out['blunder_pct']:.1f}%")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--bag", default="data/chessbench/action_value_test.bag")
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.ckpt, args.bag, args.positions, args.device)
