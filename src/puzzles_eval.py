"""Lichess puzzle accuracy for the action-value model.

    python -m src.puzzles_eval --ckpt checkpoints/chess_best.pt --num 1000

A puzzle gives a position and a forced winning line. In this file the FEN has
already been advanced past the opponent's move, so `Moves[0]` IS the first move
to find (verified against the SAN in the Solution column): the solver plays on
even plies, the opponent's forced replies on odd ones. The puzzle counts as
solved only if every solver move is found (the metric used in the ChessBench
paper, so numbers are comparable).

Scoring a position means evaluating each legal move: the model returns a
distribution over win-probability buckets for the (position, move) pair, and we
take its expected value. All legal moves of a position go through as one batch.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess
import numpy as np
import torch

from .chess_data import ACTION_OFFSET, MOVE_TO_ACTION, tokenize_fen
from .model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    model.eval().to(device)
    num_buckets = ck.get("num_buckets", ck["cfg"]["num_classes"])
    # bucket centres, matching the uniform edges used when packing the data
    centres = torch.tensor((np.arange(num_buckets) + 0.5) / num_buckets,
                           dtype=torch.float32, device=device)
    return model, centres, ck


@torch.no_grad()
def best_move(model, centres, board: chess.Board, device: str) -> str:
    """Highest expected win probability among the legal moves."""
    moves = [m.uci() for m in board.legal_moves]
    state = torch.from_numpy(tokenize_fen(board.fen()).astype(np.int64))
    x = state.repeat(len(moves), 1)
    actions = torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves], dtype=torch.long)
    x = torch.cat([x, actions[:, None]], dim=1).to(device)
    logits, _ = model(x)
    values = (logits.softmax(-1) * centres).sum(-1)
    return moves[int(values.argmax())]


def run(ckpt: str, puzzles_csv: str, num: int, device: str) -> dict:
    model, centres, ck = load_model(ckpt, device)
    solved = attempted = 0
    by_rating: dict[str, list[int]] = {}

    with open(puzzles_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if attempted >= num:
                break
            board = chess.Board(row["FEN"])
            line = row["Moves"].split()
            ok = True
            for ply, uci in enumerate(line):
                if ply % 2 == 0:            # our move: the model has to find it
                    if best_move(model, centres, board, device) != uci:
                        ok = False
                        break
                board.push(chess.Move.from_uci(uci))
            attempted += 1
            solved += ok
            band = f"{int(row['Rating']) // 500 * 500}-{int(row['Rating']) // 500 * 500 + 499}"
            by_rating.setdefault(band, []).append(int(ok))

    acc = solved / max(1, attempted)
    print(f"[puzzles] {solved}/{attempted} solved = {acc*100:.1f}%  "
          f"(checkpoint step {ck.get('step')}, val loss {ck.get('val'):.4f})")
    for band in sorted(by_rating, key=lambda b: int(b.split('-')[0])):
        v = by_rating[band]
        print(f"  rating {band:>9}: {sum(v)/len(v)*100:5.1f}%  (n={len(v)})")
    return {"accuracy": acc, "solved": solved, "attempted": attempted}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--puzzles", default="data/chessbench/puzzles.csv")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.ckpt, args.puzzles, args.num, args.device)
