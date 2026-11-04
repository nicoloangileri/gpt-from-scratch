"""Why search hurts after a puzzle fine-tune: the root ranking, or the leaves?

    python -m src.root_ranking --model 500M=a.pt --model 500M_w03=b.pt \\
        --puzzles data/chessbench/puzzles.csv --num 10000 --out root_ranking.json

On 7,616 puzzles (results/30) depth 2 adds 2.5 points to the 500M model and takes
1.1 away from its puzzle fine-tune. The reading in the README: the ranking loss
trains Q(s, a) at the position where the puzzle move is played, the root of the
search; nothing trains the values of the positions after a move, the leaves; so at
depth 2 the leaves overrule a root that is now right. This file tests it two ways.

1. Swap the networks. At depth 2 a move's value is min(Q_root(s, a), 1 - max_a'
   Q_leaf(s', a')), and nothing forces Q_root and Q_leaf to come from the same
   network. Fine-tune root with the base model's leaves: if the leaves are the
   problem, and the base model's leaves are no better, this stays where the
   fine-tune's depth 2 is. If the fine-tune damaged its own leaves, it recovers.
   The other hybrid (base root, fine-tune leaves) says what the fine-tune's leaves
   do to a root that was never fine-tuned.

2. Look at the ranks. At each puzzle's first solver move: where the solution ranks
   by Q_root alone, by the backed-up value alone, and by their minimum, per network.
   The reading predicts the fine-tune's gain is in the first and not the second.

One pass does everything. Depth-2 search at the root never prunes a depth-1 child
(a depth-1 node returns its best Q after one network call, whatever alpha and beta
say), so the depth-2 choice is exactly the argmax of the minimum above, with the
same tie order as src/search.py. Every network's root values and every network's
child values are computed once per position, and every root/leaf pairing is read
off them; the depth-1 and same-network depth-2 configurations reproduce results/30
puzzle by puzzle, which is the check that the shortcut is the search. Positions are
evaluated with the prefix shared (src/fast_eval.py), exact to float rounding.
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

from .fast_eval import network_values
from .puzzles_eval import load_model, summarise_compare

EXACT = (0.5, 1.0)


def terminal_value(board: chess.Board) -> float | None:
    """The exact value, for the side that just moved, of a position that ends the
    game, or None. The same rules, in the same order, as Searcher._q_values."""
    if board.is_checkmate():
        return 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    if board.halfmove_clock >= 100:
        return 0.5
    if board.halfmove_clock >= 4 and board.is_repetition(2):
        return 0.5
    return None


def _exact_after(board: chess.Board, moves: list[str]) -> list[float | None]:
    out = []
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        out.append(terminal_value(board))
        board.pop()
    return out


def _override(values: np.ndarray, exact: list[float | None]) -> np.ndarray:
    values = values.copy()
    for i, e in enumerate(exact):
        if e is not None:
            values[i] = e
    return values


def evaluate_node(models: dict, board: chess.Board, device: str) -> dict:
    """Root Q and backed-up child values of every legal move, for every network.

    q[name][i]  = Q(s, a_i), terminal moves exact (src/search.py _q_values)
    bu[name][i] = 1 - max_a' Q(s'_i, a'), the depth-1 value of the child backed up;
                  for a move that ends the game it is the exact value itself.
    """
    moves = [m.uci() for m in board.legal_moves]
    exact = _exact_after(board, moves)
    children, child_moves, child_exact, open_idx = [], [], [], []
    for i, uci in enumerate(moves):
        if exact[i] is not None:
            continue
        child = board.copy()
        child.push(chess.Move.from_uci(uci))
        cm = [m.uci() for m in child.legal_moves]   # not empty: no legal move is terminal
        children.append(child)
        child_moves.append(cm)
        child_exact.append(_exact_after(child, cm))
        open_idx.append(i)
    q, bu = {}, {}
    for name, (model, centres) in models.items():
        q[name] = _override(network_values(model, centres, [board], [moves], device)[0], exact)
        b = np.array([e if e is not None else np.nan for e in exact], dtype=np.float64)
        if children:
            for i, v, ex in zip(open_idx, network_values(model, centres, children, child_moves, device),
                                child_exact):
                b[i] = 1.0 - _override(v, ex).max()
        bu[name] = b
    return {"moves": moves, "q": q, "bu": bu}


def depth2_values(q_root: np.ndarray, bu_leaf: np.ndarray) -> np.ndarray:
    """A move's depth-2 value: exact if the root says so, else min(Q, backed-up)."""
    return np.array([q if q in EXACT else min(q, b) for q, b in zip(q_root, bu_leaf)])


def depth2_choice(q_root: np.ndarray, bu_leaf: np.ndarray) -> int:
    """Searcher.best_move at depth 2: the first strictly best value, in the order of
    the root's Q from high to low."""
    values = depth2_values(q_root, bu_leaf)
    best_i, best_v = int(q_root.argmax()), -1.0
    for i in np.argsort(q_root)[::-1]:
        if values[i] > best_v:
            best_i, best_v = int(i), values[i]
    return best_i


def configurations(names: list[str]) -> dict[str, tuple[str, str, int]]:
    """'name@1', 'name@2' and every hybrid 'root/leaf@2' -> (root, leaf, depth)."""
    out = {}
    for n in names:
        out[f"{n}@1"] = (n, n, 1)
        out[f"{n}@2"] = (n, n, 2)
    for r in names:
        for leaf in names:
            if r != leaf:
                out[f"{r}/{leaf}@2"] = (r, leaf, 2)
    return out


def choose(node: dict, root: str, leaf: str, depth: int) -> str:
    q = node["q"][root]
    i = int(q.argmax()) if depth == 1 else depth2_choice(q, node["bu"][leaf])
    return node["moves"][i]


def _rank(values: np.ndarray, i: int) -> int:
    return 1 + int((values > values[i]).sum())


def diagnose(node: dict, solution: str) -> dict:
    """Where the solution ranks at the root, for every network and pairing."""
    i = node["moves"].index(solution)
    out = {"n_moves": len(node["moves"]), "q_rank": {}, "bu_rank": {}, "d2_rank": {},
           "q_sol": {}, "bu_sol": {}}
    for name, q in node["q"].items():
        bu = node["bu"][name]
        out["q_rank"][name] = _rank(q, i)
        out["bu_rank"][name] = _rank(bu, i)
        out["q_sol"][name] = round(float(q[i]), 4)
        out["bu_sol"][name] = round(float(bu[i]), 4)
        for leaf, bul in node["bu"].items():
            out["d2_rank"][f"{name}/{leaf}"] = _rank(depth2_values(q, bul), i)
    return out


def solve_all(models: dict, fen: str, line: list[str], device: str,
              configs: dict) -> tuple[dict[str, int], dict | None]:
    """Every configuration on one puzzle line, as solves_line plays it.

    The configurations see the same positions for as long as they keep finding the
    solution, so one evaluation per solver position serves all that are still in.
    A different move that mates also solves the puzzle, as on Lichess."""
    board, alive, result, diag = chess.Board(fen), set(configs), {}, None
    for ply, uci in enumerate(line):
        if ply % 2 == 1 and alive:
            node = evaluate_node(models, board, device)
            if diag is None:
                diag = diagnose(node, uci)
            for key in sorted(alive):
                predicted = choose(node, *configs[key])
                if predicted != uci:
                    board.push(chess.Move.from_uci(predicted))
                    result[key] = int(board.is_checkmate())
                    board.pop()
                    alive.discard(key)
        board.push(chess.Move.from_uci(uci))
    result.update({k: 1 for k in alive})
    return {k: result[k] for k in configs}, diag


def summarise_ranks(diags: list[dict]) -> dict:
    """Share of first solver positions where the solution ranks first, and the mean
    rank, by Q, by the backed-up value and by the depth-2 value."""
    out = {}
    for field in ("q_rank", "bu_rank", "d2_rank"):
        keys = diags[0][field] if diags else {}
        out[field] = {k: {"top1": float(np.mean([d[field][k] == 1 for d in diags])),
                          "mean_rank": float(np.mean([d[field][k] for d in diags]))}
                      for k in keys}
    return out


def run(ckpts: dict[str, str], puzzles_csv: str, num: int, device: str,
        hours: float | None = None, out: str | None = None) -> dict:
    models = {}
    for name, path in ckpts.items():
        model, centres, _ = load_model(path, device)
        models[name] = (model, centres)
    configs = configurations(list(models))
    keys = list(configs)
    t0 = time.time()
    deadline = t0 + hours * 3600 if hours is not None else None
    rows, diags = [], []

    def save(final: bool = False) -> dict:
        summary = {"checkpoints": ckpts, "seconds": round(time.time() - t0), "finished": final,
                   **summarise_compare(rows, keys), "ranks": summarise_ranks(diags),
                   "rows": rows, "diagnostics": diags}
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
                print(f"[root] deadline reached after {len(rows)} puzzles", flush=True)
                break
            solved, diag = solve_all(models, row["FEN"], row["Moves"].split(), device, configs)
            rows.append({"id": row.get("PuzzleId"), "rating": int(row["Rating"]), **solved})
            diags.append({"id": row.get("PuzzleId"), **diag})
            if len(rows) % 100 == 0:
                acc = "  ".join(f"{k} {sum(r[k] for r in rows) / len(rows) * 100:.1f}%" for k in keys)
                print(f"[root] {len(rows)} puzzles, {(time.time() - t0) / 3600:.2f}h: {acc}", flush=True)
                save()
    summary = save(final=True)
    n = len(rows)
    print(f"[root] all ({n} puzzles): " + "  ".join(
        f"{k} {v['accuracy'] * 100:.1f}%" for k, v in summary["all"].items()), flush=True)
    for field, per in summary["ranks"].items():
        print(f"[root] {field}: " + "  ".join(
            f"{k} top-1 {v['top1'] * 100:.1f}% mean {v['mean_rank']:.2f}" for k, v in per.items()),
            flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--puzzles", default="data/chessbench/puzzles.csv")
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    run(dict(m.split("=", 1) for m in args.model), args.puzzles, args.num, args.device,
        args.hours, args.out)
