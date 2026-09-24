"""Move quality and calibration on ChessBench's held-out test positions.

    python -m src.agreement_eval --ckpt checkpoints/chess_225M.pt --device mps

Puzzle accuracy is all-or-nothing over a whole forced line, which makes it a
coarse instrument. This measures the model move by move, on every fully annotated
position of the action-value test shard (61,833 positions, ~30 legal moves each),
with the two ranking metrics the ChessBench paper reports in its ablations and a
calibration check it does not:

- action accuracy: the model's move is Stockfish's best move. Reported twice. The
  paper's definition compares against the argmax; but win probability saturates,
  and in 5.9% of these positions several moves share Stockfish's top value -- there
  the argmax is an arbitrary pick and a different, equally winning move is scored
  as a miss. So "picks A best move" is reported next to it.
- Kendall's tau-b between the model's values and Stockfish's over all legal moves
  of a position, averaged over positions: does it order the moves the same way?
- win probability given up per move, and the share of moves giving up > 15 points.
- calibration of the value itself: when the model says 0.7, is Stockfish's number
  0.7? (binned error, ECE) -- and of its whole distribution: does the truth fall
  inside the central 50% / 90% of the predicted buckets 50% / 90% of the time?

The shard stores one record per (position, legal move), shuffled; grouping the file
by FEN puts each position back together with every move scored.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import chess
import numpy as np
import torch

from .chess_data import (ACTION_OFFSET, MOVE_TO_ACTION, iter_action_value, tokenize_fen,
                         win_prob_to_bucket)
from .puzzles_eval import load_model

BLUNDER = 0.15


def kendall_tau_b(x, y) -> float:
    """Kendall's tau-b: rank agreement in [-1, 1], with ties counted correctly.

    Ties matter here: Stockfish's win probabilities saturate, so several moves of a
    won position share a value. NaN when either side has no strict ordering at all.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    i, j = np.triu_indices(len(x), k=1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    den = np.sqrt(float(np.count_nonzero(sx)) * float(np.count_nonzero(sy)))
    return float((sx * sy).sum() / den) if den else float("nan")


def mid_pit(probs: np.ndarray, bucket: int) -> float:
    """Where the truth lands in the predicted distribution: F(b - 1) + p(b) / 2.

    For a calibrated model this is uniform on [0, 1]. The half-weight on the truth's
    own bucket is the standard correction for a discrete distribution; without it,
    64 buckets would make every interval look a little too wide.
    """
    return float(probs[:bucket].sum() + 0.5 * probs[bucket])


def mid_pits(probs: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    """mid_pit for every row at once. The per-move Python loop it replaces ran 1.84M
    times per model and was most of the evaluation's time -- and of the laptop's heat."""
    rows = np.arange(len(buckets))
    own = probs[rows, buckets]
    return np.cumsum(probs, axis=1)[rows, buckets] - own + 0.5 * own


def position_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Everything measured on one position, from the model's and Stockfish's values."""
    chosen = int(predicted.argmax())
    best = float(truth.max())
    return {
        "strict": chosen == int(truth.argmax()),
        "any_best": bool(truth[chosen] >= best - 1e-9),
        "top3": chosen in set(np.argsort(truth)[::-1][:3].tolist()),
        "given_up": best - float(truth[chosen]),
        "tau": kendall_tau_b(predicted, truth),
    }


def calibration(pred: np.ndarray, truth: np.ndarray, pit: np.ndarray, bins: int = 10) -> dict:
    """Binned value calibration and interval coverage of the predicted distribution."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(pred, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        gap = abs(float(pred[m].mean()) - float(truth[m].mean()))
        ece += m.mean() * gap
        table.append((edges[b], edges[b + 1], int(m.sum()), float(pred[m].mean()),
                      float(truth[m].mean())))
    centred = np.abs(pit - 0.5)
    return {"mae": float(np.abs(pred - truth).mean()), "ece": float(ece),
            "cover50": float((centred <= 0.25).mean()),
            "cover90": float((centred <= 0.45).mean()), "table": table}


def complete_positions(bag: str) -> list[tuple[str, list[str], np.ndarray]]:
    """(fen, moves, Stockfish win probabilities) for every position whose legal
    moves are all annotated -- a partial one would make both ranking metrics lie."""
    groups: dict[str, list] = defaultdict(list)
    for fen, move, win_prob in iter_action_value(bag):
        groups[fen].append((move, win_prob))
    out = []
    for fen, records in groups.items():
        legal = {m.uci() for m in chess.Board(fen).legal_moves}
        moves = [m for m, _ in records]
        if set(moves) == legal and len(moves) >= 2:
            out.append((fen, moves, np.array([p for _, p in records])))
    return out


@torch.no_grad()
def _score(model, centres, items, device):
    """Several positions in one forward pass: -> [(values, probs)] per position."""
    xs, sizes = [], []
    for fen, moves, _ in items:
        state = torch.from_numpy(tokenize_fen(fen).astype(np.int64))
        actions = torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves])
        xs.append(torch.cat([state.repeat(len(moves), 1), actions[:, None]], dim=1))
        sizes.append(len(moves))
    logits, _ = model(torch.cat(xs).to(device))
    probs = logits.float().softmax(-1)
    values = (probs * centres).sum(-1)
    probs, values = probs.cpu().numpy(), values.cpu().numpy()
    out, k = [], 0
    for n in sizes:
        out.append((values[k:k + n], probs[k:k + n]))
        k += n
    return out


def run(ckpt: str, bag: str, positions: int | None, device: str,
        per_batch: int = 64) -> dict:
    items = complete_positions(bag)
    if positions:
        items = items[:positions]
    model, centres, ck = load_model(ckpt, device)
    num_buckets = len(centres)

    per_position, preds, truths, pits = [], [], [], []
    for start in range(0, len(items), per_batch):
        chunk = items[start:start + per_batch]
        for (fen, moves, truth), (values, probs) in zip(chunk, _score(model, centres, chunk, device)):
            per_position.append(position_metrics(values, truth))
            buckets = win_prob_to_bucket(truth, num_buckets)
            preds.append(values)
            truths.append(truth)
            pits.append(mid_pits(probs, buckets.astype(np.int64)))

    pp = per_position
    taus = np.array([m["tau"] for m in pp])
    given_up = np.array([m["given_up"] for m in pp])
    cal = calibration(np.concatenate(preds), np.concatenate(truths), np.concatenate(pits))
    out = {
        "checkpoint_step": ck.get("step"), "positions": len(pp),
        "moves_scored": int(sum(len(t) for t in truths)),
        "action_acc": 100 * np.mean([m["strict"] for m in pp]),
        "any_best_acc": 100 * np.mean([m["any_best"] for m in pp]),
        "top3": 100 * np.mean([m["top3"] for m in pp]),
        "kendall_tau": float(np.nanmean(taus)),
        "tau_undefined": int(np.isnan(taus).sum()),
        "mean_given_up": float(given_up.mean()), "median_given_up": float(np.median(given_up)),
        "blunder_pct": 100 * float((given_up > BLUNDER).mean()),
        **{k: v for k, v in cal.items() if k != "table"},
    }
    print(f"[agreement] {ckpt} (step {out['checkpoint_step']}): {out['positions']:,} positions, "
          f"{out['moves_scored']:,} moves")
    print(f"  action accuracy (paper: = argmax)  : {out['action_acc']:.1f}%")
    print(f"  picks a move of maximal value      : {out['any_best_acc']:.1f}%")
    print(f"  one of Stockfish's top 3           : {out['top3']:.1f}%")
    print(f"  Kendall's tau-b, mean over positions: {out['kendall_tau']:.3f} "
          f"({out['tau_undefined']} positions undefined)")
    print(f"  win probability given up           : mean {out['mean_given_up']:.4f}, "
          f"median {out['median_given_up']:.4f}")
    print(f"  moves giving up > {BLUNDER:.2f}           : {out['blunder_pct']:.1f}%")
    print(f"  value calibration                  : MAE {out['mae']:.4f}, ECE {out['ece']:.4f}")
    print(f"  truth inside central 50% / 90%     : {100 * out['cover50']:.1f}% / "
          f"{100 * out['cover90']:.1f}%")
    print("  reliability (predicted bin: n, mean predicted, mean Stockfish)")
    for lo, hi, n, p, t in cal["table"]:
        print(f"    [{lo:.1f}, {hi:.1f}) n={n:>9,}  {p:.3f}  {t:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--bag", default="data/chessbench/action_value_test.bag")
    ap.add_argument("--positions", type=int, default=None, help="default: all of them")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.ckpt, args.bag, args.positions, args.device)
