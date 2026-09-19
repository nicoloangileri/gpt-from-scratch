"""Lichess puzzle accuracy for the action-value model.

    python -m src.puzzles_eval --ckpt checkpoints/chess_best.pt --num 1000

A puzzle gives a position and a forced winning line, in raw Lichess format: the
FEN is the position BEFORE the opponent's move, `Moves[0]` is that opponent's
move (usually the mistake that allows the tactic), and the solver plays on the
odd plies. The puzzle counts as solved only if every solver move is found, or if
a different move also mates in one -- Lichess accepts any mate. This matches
DeepMind's searchless_chess/src/puzzles.py, so numbers are comparable.

History, because the mistake was made here once: ChessBench's puzzles.csv is NOT
pre-advanced. Its FEN and Moves are byte-identical to lichess_db_puzzle for the
same PuzzleId (checked on 9,995 of the 10,000 test puzzles). A 9/9 "fix" read
Moves[0] as the solver's move after checking it against the Solution column --
but that column is simply the SAN of the whole Moves sequence, so the check was a
tautology, and every puzzle accuracy reported from then until 19/9 asked the model
to predict the opponent's moves instead of the solution.

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
def best_move(model, centres, board: chess.Board, device: str,
              score_draws: bool = True) -> str:
    """Highest expected win probability among the legal moves.

    The model predicts win probability, which saturates: in a won position every
    sane move is worth ~0.99, so shuffling a rook and delivering mate look
    identical and the model repeats until the game is drawn. It measured 21 points
    of material up and still drew.

    `score_draws` fixes that at its root rather than with a heuristic: a move that
    forces a repetition, stalemate or the fifty-move rule is worth exactly half a
    point, so its value is set to 0.5 instead of whatever the model imagined. When
    winning, the model then avoids it; when losing, it steers into it. Checkmate is
    scored 1.0 for the same reason — the model has no mate-distance signal to tell
    it that mate now beats mate later.
    """
    moves = [m.uci() for m in board.legal_moves]
    if not moves:
        raise ValueError("no legal moves: the game is already over")
    state = torch.from_numpy(tokenize_fen(board.fen()).astype(np.int64))
    x = state.repeat(len(moves), 1)
    actions = torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves], dtype=torch.long)
    x = torch.cat([x, actions[:, None]], dim=1).to(device)
    logits, _ = model(x)
    values = (logits.softmax(-1) * centres).sum(-1)

    if score_draws:
        for i, uci in enumerate(moves):
            board.push(chess.Move.from_uci(uci))
        # is_repetition() and can_claim_fifty_moves() both walk the move stack, and
        # this runs for every legal move of every node -- on a long game that is
        # quadratic, and it cost 116s per move in a match. A repetition needs at
        # least four halfmoves without a capture or pawn move, so the clock gates
        # the expensive check; the fifty-move rule is just the clock itself.
            if board.is_checkmate():
                values[i] = 1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                values[i] = 0.5
            elif board.halfmove_clock >= 100:
                values[i] = 0.5
            elif board.halfmove_clock >= 4 and board.is_repetition(2):
                values[i] = 0.5
            board.pop()

    return moves[int(values.argmax())]


def solves_line(board: chess.Board, moves: list[str], pick) -> bool:
    """True if `pick` finds every solver move of a raw-Lichess puzzle line.

    Ply 0 is the opponent's move and is simply played; the solver moves on odd
    plies. A different move that mates in one also solves it, as on Lichess. Same
    logic as DeepMind's evaluate_puzzle_from_board.
    """
    for ply, uci in enumerate(moves):
        if ply % 2 == 1:
            predicted = pick(board)
            if predicted != uci:
                board.push(chess.Move.from_uci(predicted))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(uci))
    return True


def run(ckpt: str, puzzles_csv: str, num: int, device: str, depth: int = 1) -> dict:
    """depth > 1 puts alpha-beta search on top of the network (src/search.py)."""
    model, centres, ck = load_model(ckpt, device)
    if depth > 1:
        from .search import Searcher
        pick = lambda board: Searcher(model, centres, device).best_move(board, depth)
    else:
        pick = lambda board: best_move(model, centres, board, device)
    solved = attempted = 0
    by_rating: dict[str, list[int]] = {}

    with open(puzzles_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if attempted >= num:
                break
            ok = solves_line(chess.Board(row["FEN"]), row["Moves"].split(), pick)
            attempted += 1
            solved += ok
            band = f"{int(row['Rating']) // 500 * 500}-{int(row['Rating']) // 500 * 500 + 499}"
            by_rating.setdefault(band, []).append(int(ok))

    acc = solved / max(1, attempted)
    print(f"[puzzles] depth {depth}: {solved}/{attempted} solved = {acc*100:.1f}%  "
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
    ap.add_argument("--depth", type=int, default=1, help="1 = network alone, 2+ = search")
    args = ap.parse_args()
    run(args.ckpt, args.puzzles, args.num, args.device, args.depth)
