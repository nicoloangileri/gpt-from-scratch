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
import json
import time
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


def picker(model, centres, device: str, depth: int):
    """The move chooser for a depth: the network alone at 1, alpha-beta on top beyond."""
    if depth > 1:
        from .search import Searcher
        return lambda board: Searcher(model, centres, device).best_move(board, depth)
    return lambda board: best_move(model, centres, board, device)


def run(ckpt: str, puzzles_csv: str, num: int, device: str, depth: int = 1) -> dict:
    """depth > 1 puts alpha-beta search on top of the network (src/search.py)."""
    model, centres, ck = load_model(ckpt, device)
    pick = picker(model, centres, device, depth)
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


def summarise_compare(rows: list[dict], keys: list[str]) -> dict:
    """Accuracy per checkpoint@depth on the first 1,000 puzzles -- the slice every earlier
    search figure used -- and on all of them, plus depth 2 minus depth 1 per checkpoint,
    paired by puzzle: the noise of a difference between two configurations that saw the
    same puzzles is that of the puzzles where they disagree, not of two separate samples."""
    out: dict = {"attempted": len(rows)}
    for part_name, part in (("first_1000", rows[:1000]), ("all", rows)):
        out[part_name] = {k: {"solved": sum(r[k] for r in part),
                              "accuracy": sum(r[k] for r in part) / max(1, len(part))}
                          for k in keys}
    bands: dict = {}
    for r in rows:
        lo = r["rating"] // 500 * 500
        bands.setdefault(f"{lo}-{lo + 499}", []).append(r)
    out["all_by_rating"] = {
        band: {"n": len(v), **{k: sum(r[k] for r in v) / len(v) for k in keys}}
        for band, v in sorted(bands.items(), key=lambda kv: int(kv[0].split("-")[0]))}
    out["depth2_minus_depth1"] = {}
    for name in sorted({k.split("@")[0] for k in keys}):
        if f"{name}@1" not in keys or f"{name}@2" not in keys or len(rows) < 2:
            continue
        d = np.array([r[f"{name}@2"] - r[f"{name}@1"] for r in rows], dtype=np.float64)
        out["depth2_minus_depth1"][name] = {
            "points": float(d.mean() * 100), "se_points": float(d.std(ddof=1) / np.sqrt(len(d)) * 100),
            "only_depth2_solves": int((d > 0).sum()), "only_depth1_solves": int((d < 0).sum())}
    return out


def compare_search(ckpts: dict[str, str], puzzles_csv: str, num: int, device: str,
                   depths=(1, 2), hours: float | None = None, out: str | None = None,
                   picks: dict | None = None) -> dict:
    """Every checkpoint at every depth on the same puzzles, one puzzle at a time.

    The search figures so far are 1,000 puzzles per point, a point or two of noise
    each, and the trend they suggest has a point that does not fit. Here every
    checkpoint@depth answers a puzzle before the next one starts, so the comparisons
    are paired, and a deadline cuts the list without leaving one configuration ahead of
    another. The picks are run()'s, so the first 1,000 reproduce its numbers. The
    output is rewritten every 100 puzzles: a session killed at its limit keeps them.
    """
    if picks is None:
        picks = {}
        for name, path in ckpts.items():
            model, centres, _ = load_model(path, device)
            for d in depths:
                picks[f"{name}@{d}"] = picker(model, centres, device, d)
    keys = list(picks)
    t0 = time.time()
    deadline = t0 + hours * 3600 if hours is not None else None
    rows: list[dict] = []

    def save(final: bool = False) -> dict:
        summary = {"checkpoints": ckpts, "depths": list(depths), "seconds": round(time.time() - t0),
                   "finished": final, **summarise_compare(rows, keys), "rows": rows}
        if out:
            tmp = Path(out).with_suffix(".tmp")
            tmp.write_text(json.dumps(summary) + "\n")
            tmp.replace(out)
        return summary

    with open(puzzles_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if len(rows) >= num:
                break
            if deadline is not None and time.time() > deadline:
                print(f"[compare] deadline reached after {len(rows)} puzzles", flush=True)
                break
            moves = row["Moves"].split()
            rows.append({"id": row.get("PuzzleId"), "rating": int(row["Rating"]),
                         **{k: int(solves_line(chess.Board(row["FEN"]), moves, pick))
                            for k, pick in picks.items()}})
            if len(rows) % 100 == 0:
                acc = "  ".join(f"{k} {sum(r[k] for r in rows) / len(rows) * 100:.1f}%" for k in keys)
                print(f"[compare] {len(rows)} puzzles, {(time.time() - t0) / 3600:.2f}h: {acc}", flush=True)
                save()
    summary = save(final=True)
    for part in ("first_1000", "all"):
        n = min(len(rows), 1000) if part == "first_1000" else len(rows)
        print(f"[compare] {part} ({n} puzzles): " + "  ".join(
            f"{k} {v['solved']}/{n} = {v['accuracy'] * 100:.1f}%" for k, v in summary[part].items()), flush=True)
    for name, v in summary["depth2_minus_depth1"].items():
        print(f"[compare] {name}: depth 2 - depth 1 = {v['points']:+.2f} points (se {v['se_points']:.2f}), "
              f"{v['only_depth2_solves']} solved only at depth 2, {v['only_depth1_solves']} only at depth 1",
              flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--puzzles", default="data/chessbench/puzzles.csv")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth", type=int, default=1, help="1 = network alone, 2+ = search")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=CKPT",
                    help="repeatable: every checkpoint at every --depths, paired by puzzle")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--hours", type=float, default=None, help="--compare stops cleanly after this")
    ap.add_argument("--out", default=None, help="--compare's JSON, rewritten every 100 puzzles")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.compare:
        compare_search(dict(c.split("=", 1) for c in args.compare), args.puzzles, args.num,
                       args.device, args.depths, args.hours, args.out)
    else:
        run(args.ckpt, args.puzzles, args.num, args.device, args.depth)
