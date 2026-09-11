"""Does the model actually play chess? Games against calibrated baselines.

    python -m src.play_eval --ckpt checkpoints/chess_best.pt --games 20

Puzzle accuracy measures whether the model finds a forced tactic. Playing a game
is a different test: forty moves without giving material away. A model can score
well on puzzles and still lose to a random mover, so this runs real games against
two opponents whose strength is known by construction:

  random    — uniform over legal moves. Losing to this means something is broken.
  greedy    — takes the most valuable capture available, else a random move.
              Roughly a beginner who only sees one ply, and the real bar: beating
              it means the model understands material, not just tactics.
"""

from __future__ import annotations

import argparse
import random

import chess
import torch

from .puzzles_eval import best_move, load_model

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def random_mover(board: chess.Board, rng: random.Random) -> chess.Move:
    return rng.choice(list(board.legal_moves))


def greedy_mover(board: chess.Board, rng: random.Random) -> chess.Move:
    """Best immediate capture by piece value; ties and quiet positions go random."""
    best, best_gain = [], 0
    for move in board.legal_moves:
        gain = 0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            gain = PIECE_VALUE[chess.PAWN] if captured is None else PIECE_VALUE[captured.piece_type]
        if gain > best_gain:
            best, best_gain = [move], gain
        elif gain == best_gain:
            best.append(move)
    return rng.choice(best if best else list(board.legal_moves))


def material(board: chess.Board, colour: bool) -> int:
    total = 0
    for square, piece in board.piece_map().items():
        total += PIECE_VALUE[piece.piece_type] * (1 if piece.color == colour else -1)
    return total


def play_game(model, centres, device, opponent, model_is_white: bool, rng, max_moves=120):
    """Returns 1 win, 0.5 draw, 0 loss (from the model's side), plus stats."""
    board = chess.Board()
    failures = 0
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        if board.turn == model_is_white:
            try:
                uci = best_move(model, centres, board, device)
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move {uci}")
            except Exception:
                failures += 1
                move = random_mover(board, rng)
        else:
            move = opponent(board, rng)
        board.push(move)

    colour = chess.WHITE if model_is_white else chess.BLACK
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome.winner is None:
            score = 0.5
        else:
            score = 1.0 if outcome.winner == colour else 0.0
        reason = outcome.termination.name
    else:                                    # hit the move cap: adjudicate on material
        edge = material(board, colour)
        score = 1.0 if edge > 1 else 0.0 if edge < -1 else 0.5
        reason = f"ADJUDICATED({edge:+d})"
    return score, board.fullmove_number, failures, reason


def run(ckpt: str, games: int, device: str, seed: int = 0) -> dict:
    model, centres, ck = load_model(ckpt, device)
    rng = random.Random(seed)
    results = {}
    for name, opponent in (("random", random_mover), ("greedy material", greedy_mover)):
        wins = draws = losses = failures = 0
        lengths, reasons = [], {}
        for i in range(games):
            score, length, fails, reason = play_game(
                model, centres, device, opponent, model_is_white=(i % 2 == 0), rng=rng)
            wins += score == 1.0
            draws += score == 0.5
            losses += score == 0.0
            failures += fails
            lengths.append(length)
            reasons[reason] = reasons.get(reason, 0) + 1
        pct = (wins + 0.5 * draws) / games
        print(f"[play] vs {name:16} {wins}W-{draws}D-{losses}L  score {pct*100:.0f}%  "
              f"| avg {sum(lengths)/len(lengths):.0f} moves | illegal {failures}")
        print(f"       endings: {reasons}")
        results[name] = {"w": wins, "d": draws, "l": losses, "score": pct}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.ckpt, args.games, args.device, args.seed)
