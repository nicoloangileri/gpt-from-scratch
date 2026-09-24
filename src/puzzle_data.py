"""Lichess puzzles as a ranking task, to teach the model to FIND tactics.

    python -m src.puzzle_data --csv data/lichess/lichess_db_puzzle.csv.zst \
        --exclude data/chessbench/puzzles.csv --out data/puzzles --max_puzzles 200000

The action-value model learns how good a move is, which is not the same as knowing
which move to play: the 225M-example model picks Stockfish's first choice on 45% of
the test positions while giving up only 2.3 points of win probability per move on
average (src/agreement_eval.py, full test shard). It plays reasonably and misses
tactics. Lichess publishes 6,100,952 puzzles (CC0)
whose label is exactly the missing one -- "this move, not the others" -- so they are
packed here as a ranking problem: one position, the correct move, and a sample of the
legal moves it has to beat.

Format, matching the raw Lichess CSV that ChessBench's puzzles.csv copies verbatim:
the FEN is the position BEFORE the opponent's move, `Moves[0]` is that move, and the
solver plays the ODD plies. Every solver ply becomes one training example.

The 10,000 test puzzles are excluded by PuzzleId. Without that, a fine-tune would be
training on the benchmark and every puzzle number in this repo would become
meaningless -- the files are byte-identical for the same id.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

import chess
import numpy as np

from .chess_data import MOVE_TO_ACTION, SEQUENCE_LENGTH, tokenize_fen

NUM_CANDIDATES = 16          # the correct move plus up to 15 legal alternatives
PAD_ACTION = 0               # never read: `ncand` says how many entries are real


def held_out_puzzle_ids(path: str | Path) -> set[str]:
    """The PuzzleIds of the held-out set, to keep them out of training."""
    with open(path, newline="") as fh:
        return {row["PuzzleId"] for row in csv.DictReader(fh)}


def iter_puzzle_rows(path: str | Path, limit: int | None = None):
    """Stream the zstd-compressed Lichess puzzle CSV without unpacking it to disk."""
    import zstandard

    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        for i, row in enumerate(csv.DictReader(text)):
            if limit is not None and i >= limit:
                return
            yield row


def solver_positions(fen: str, moves: list[str]):
    """Yield (board, correct_uci) for every ply the solver has to play.

    Ply 0 is the opponent's move -- the mistake that allows the tactic -- so the
    solver moves on the odd plies. This is the same convention as
    src/puzzles_eval.py and DeepMind's searchless_chess/src/puzzles.py; getting it
    backwards once invalidated ten days of measurements.
    """
    board = chess.Board(fen)
    for ply, uci in enumerate(moves):
        if ply % 2 == 1:
            yield board.copy(stack=False), uci
        board.push(chess.Move.from_uci(uci))


def candidates(board: chess.Board, correct: str, num: int,
               rng: random.Random) -> list[int] | None:
    """Action indices: the correct move first, then a sample of its alternatives.

    Returns None when the position cannot be used: no alternatives to rank against.
    The action-space checks are defensive only -- all 1968 UCI moves are encoded,
    underpromotions and castling included (tests/test_chess_data.py).

    Mating alternatives are dropped when the correct move is itself a mate: Lichess
    accepts any mate in one, so scoring another one is not an error and punishing it
    would teach the model something false.
    """
    correct_move = chess.Move.from_uci(correct)
    if correct not in MOVE_TO_ACTION:
        return None
    others = [m for m in board.legal_moves if m != correct_move]
    if not others:
        return None                       # only one legal move: nothing to learn

    board.push(correct_move)
    correct_mates = board.is_checkmate()
    board.pop()
    if correct_mates:
        kept = []
        for m in others:
            board.push(m)
            mates = board.is_checkmate()
            board.pop()
            if not mates:
                kept.append(m)
        others = kept
        if not others:
            return None                   # every move mates: nothing to learn

    others = [m for m in others if m.uci() in MOVE_TO_ACTION]
    if not others:
        return None
    if len(others) > num - 1:
        others = rng.sample(others, num - 1)
    return [MOVE_TO_ACTION[correct]] + [MOVE_TO_ACTION[m.uci()] for m in others]


def pack_puzzles(csv_path: str | Path, out_dir: str | Path,
                 exclude_ids: set[str] | None = None,
                 max_puzzles: int | None = None, val_positions: int = 20_000,
                 num_candidates: int = NUM_CANDIDATES, seed: int = 0,
                 min_rating: int = 0, log_every: int = 200_000) -> dict:
    """Write {split}_states.bin, {split}_cands.bin and {split}_ncand.bin.

    States are uint8 (N, 77), candidates uint16 (N, num_candidates) with the correct
    move in column 0, and ncand uint8 (N,) the number of real entries in each row.
    Validation comes from the FIRST puzzles in the file rather than a random sample,
    so a packing run that is cut short still has a usable split.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)

    handles = {}
    for split in ("val", "train"):
        handles[split] = {
            "states": open(out / f"{split}_states.bin", "wb"),
            "cands": open(out / f"{split}_cands.bin", "wb"),
            "ncand": open(out / f"{split}_ncand.bin", "wb"),
        }

    counts = {"val": 0, "train": 0}
    seen = skipped_excluded = skipped_unusable = 0
    for row in iter_puzzle_rows(csv_path, limit=max_puzzles):
        seen += 1
        if row["PuzzleId"] in exclude_ids:
            skipped_excluded += 1
            continue
        if min_rating and int(row["Rating"]) < min_rating:
            continue
        moves = row["Moves"].split()
        try:
            positions = list(solver_positions(row["FEN"], moves))
        except ValueError:
            skipped_unusable += 1        # malformed FEN or move
            continue
        for board, correct in positions:
            cand = candidates(board, correct, num_candidates, rng)
            if cand is None:
                skipped_unusable += 1
                continue
            split = "val" if counts["val"] < val_positions else "train"
            padded = cand + [PAD_ACTION] * (num_candidates - len(cand))
            h = handles[split]
            h["states"].write(tokenize_fen(board.fen()).astype(np.uint8).tobytes())
            h["cands"].write(np.asarray(padded, dtype=np.uint16).tobytes())
            h["ncand"].write(np.uint8(len(cand)).tobytes())
            counts[split] += 1
        if log_every and seen % log_every == 0:
            print(f"[puzzles] {seen:,} puzzles -> {counts['train']:,} train "
                  f"+ {counts['val']:,} val positions", flush=True)

    for h in handles.values():
        for fh in h.values():
            fh.close()
    stats = {"puzzles_read": seen, "excluded": skipped_excluded,
             "unusable_positions": skipped_unusable, **counts,
             "num_candidates": num_candidates}
    print(f"[puzzles] done: {stats}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/lichess/lichess_db_puzzle.csv.zst")
    ap.add_argument("--exclude", default="data/chessbench/puzzles.csv",
                    help="the held-out puzzles, excluded by PuzzleId")
    ap.add_argument("--out", default="data/puzzles")
    ap.add_argument("--max_puzzles", type=int, default=None)
    ap.add_argument("--val_positions", type=int, default=20_000)
    ap.add_argument("--min_rating", type=int, default=0)
    args = ap.parse_args()

    exclude = held_out_puzzle_ids(args.exclude) if args.exclude else set()
    print(f"[puzzles] excluding {len(exclude):,} held-out ids", flush=True)
    pack_puzzles(args.csv, args.out, exclude, args.max_puzzles,
                 args.val_positions, min_rating=args.min_rating)


if __name__ == "__main__":
    main()
