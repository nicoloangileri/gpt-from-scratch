"""Two checkpoints against each other: which one actually plays better?

    python -m src.match --a checkpoints/chess_225M.pt --b checkpoints/chess_225M_puzzles.pt \
        --depth 1 --positions 200

Lichess answers this slowly and noisily: a rating against a pool of bots that
changes with the time of day needs hundreds of games per model to separate two
close ones. A direct match answers it in one evening, the way engine testers do:

- Starting positions are balanced middlegame positions from real games, taken from
  ChessBench's held-out test shard -- Stockfish gives the side to move between 45%
  and 60%, so neither model starts from a won or lost position, and neither has
  trained on them. Varied starts matter: two deterministic players from the same
  position play the same game every time (the first "6-0" in this project was two
  games played three times).
- Every position is played twice with colours swapped, so the side-to-move edge of
  a position cancels out instead of landing on whoever drew it.
- The result is an Elo difference with a 95% interval, not just a score.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import chess
import numpy as np


def elo_from_score(score: float) -> float:
    """Elo difference implied by an expected score (0.5 -> 0, 0.76 -> ~+200)."""
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarise(results: list[float]) -> dict:
    """Score, Elo and a 95% interval from per-game results for player A (1, 0.5, 0)."""
    x = np.asarray(results, dtype=np.float64)
    n = len(x)
    if n == 0:                        # a deadline hit before the first position
        return {"games": 0, "wins": 0, "draws": 0, "losses": 0, "score": None,
                "elo": None, "elo_low": None, "elo_high": None}
    s = float(x.mean())
    se = float(x.std(ddof=0) / math.sqrt(n)) if n > 1 else 0.5
    lo, hi = s - 1.96 * se, s + 1.96 * se
    return {"games": n, "wins": int((x == 1).sum()), "draws": int((x == 0.5).sum()),
            "losses": int((x == 0).sum()), "score": s, "elo": elo_from_score(s),
            "elo_low": elo_from_score(lo), "elo_high": elo_from_score(hi)}


def balanced_positions(bag: str, count: int, seed: int = 0,
                       lo: float = 0.45, hi: float = 0.60) -> list[str]:
    """Middlegame positions from the test shard where the side to move is near even."""
    from .agreement_eval import complete_positions
    pool = []
    for fen, _, truth in complete_positions(bag):
        board = chess.Board(fen)
        if lo <= float(truth.max()) <= hi and 8 <= board.fullmove_number <= 30 \
                and len(board.piece_map()) >= 20:
            pool.append(fen)
    pool.sort()                                   # file order must not decide the sample
    random.Random(seed).shuffle(pool)
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} balanced positions, {count} requested")
    return pool[:count]


def play(fen: str, white, black, max_plies: int = 300) -> float:
    """One game from `fen`; returns white's result. Unfinished games are draws."""
    board = chess.Board(fen)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        mover = white if board.turn == chess.WHITE else black
        board.push(chess.Move.from_uci(mover(board)))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def decided(results: list[float], first_to: float) -> str | None:
    """'A' or 'B' once one side has `first_to` points and leads; None otherwise.
    Level at the target (6.5-6.5 then a draw) is not a win: play on until one leads."""
    a = sum(results)
    b = len(results) - a
    if max(a, b) >= first_to and a != b:
        return "A" if a > b else "B"
    return None


def run_match(positions: list[str], a, b, log=print, max_plies: int = 300,
              first_to: float | None = None, on_position=None,
              deadline: float | None = None) -> list[float]:
    """A plays each position once as white and once as black. Results from A's side.

    With `first_to`, the match stops at the first game after which one side has that
    many points (win 1, draw 1/2) and leads -- a first-to-7 match, like a set in
    tennis. Colours still alternate game by game.

    `on_position(results)` runs after every position, so a long match can save as it
    goes; `deadline` (a time.time() value) stops it cleanly between positions, never
    between the two colours of one, so the colour balance survives a cut."""
    results = []
    for i, fen in enumerate(positions):
        if deadline is not None and time.time() > deadline:
            log(f"[match] deadline reached after {i} positions")
            break
        for a_is_white in (True, False):
            if a_is_white:
                results.append(play(fen, a, b, max_plies))
            else:
                results.append(1.0 - play(fen, b, a, max_plies))
            if first_to is not None:
                pa = sum(results)
                log(f"[match] game {len(results)}: A {pa:g} - B {len(results) - pa:g}")
                if decided(results, first_to):
                    return results
        if on_position is not None:
            on_position(results)
        s = summarise(results)
        log(f"[match] {i + 1}/{len(positions)} positions: A {s['wins']}W {s['draws']}D "
            f"{s['losses']}L, score {s['score']:.3f}, Elo {s['elo']:+.0f} "
            f"[{s['elo_low']:+.0f}, {s['elo_high']:+.0f}]")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (results are from its side)")
    ap.add_argument("--b", required=True, help="checkpoint B")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--positions", type=int, default=100, help="each is played twice")
    ap.add_argument("--bag", default="data/chessbench/action_value_test.bag")
    ap.add_argument("--threads", type=int, default=2, help="keep the laptop usable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first-to", dest="first_to", type=float, default=None,
                    help="stop when one side reaches this many points and leads")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop cleanly after this long; results so far are kept")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from .puzzles_eval import load_model
    from .search import make_player

    positions = balanced_positions(args.bag, args.positions, args.seed)
    players = []
    for ckpt in (args.a, args.b):
        model, centres, _ = load_model(ckpt, args.device)
        players.append(make_player(model, centres, args.device, args.depth))
    print(f"[match] A = {args.a}\n[match] B = {args.b}\n[match] depth {args.depth}, "
          f"{len(positions)} positions x 2 colours, {args.threads} threads", flush=True)

    t = time.time()

    def report(results):
        summary = {"a": args.a, "b": args.b, "depth": args.depth, "seed": args.seed,
                   "seconds": round(time.time() - t), **summarise(results)}
        if args.out:                  # rewritten after every position: a cut keeps them
            Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
        return summary

    results = run_match(positions, *players, log=lambda m: print(m, flush=True),
                        first_to=args.first_to, on_position=report,
                        deadline=t + args.hours * 3600 if args.hours else None)
    summary = report(results)
    if args.first_to is not None:
        summary["first_to"] = args.first_to
        summary["winner"] = decided(results, args.first_to)   # None: ran out of positions
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
