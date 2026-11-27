"""Monte Carlo tree search guided by the action-value network.

    python -m src.mcts --ckpt checkpoints/chess_1B_w03.pt --leaf_ckpt checkpoints/chess_1B.pt \
        --sims 32 --puzzles data/chessbench/puzzles.csv --num 200

Alpha-beta at depth 2 is worth +165 Elo over the bare 1B model in games
(results/38), but a third ply at fixed depth was worth 15 (-8 to 37, results/34)
for 76 times the network calls. Fixed depth spends the same effort on every line;
a tree search spends it where the network is unsure. No random rollouts: every
expansion is one network call, which returns Q(s, a) for all the legal moves of
the new position in one batch, and that is all the tree needs.

- The prior of a move is a softmax of the network's Q over the legal moves, at
  temperature `tau` (values are win probabilities, so differences are small).
- A move not yet visited is worth the network's Q(s, a) itself: the first-play
  value is the network's own estimate of the position after the move, not a guess.
- A leaf is worth max_a Q(leaf, a) for the side to move there, the same leaf value
  as src/search.py, and the value backed up the path flips as 1 - v.
- A visited move is worth min(Q(s, a), mean backed-up value) when `pessimistic`:
  the reason src/search.py takes the min (its docstring has the measured case) is
  that a maximiser hunts for the evaluator's mistakes, and a tree search maximises
  too.

Selection is PUCT, value + c * P * sqrt(N) / (1 + n), as in AlphaZero. The move
played is the most visited, ties going to the higher value. The root is scored by
`model`, every other node by `leaf_model`, as in the bot's hybrid.

Terminal positions are scored exactly, by the rules src/root_ranking.py uses.
"""
from __future__ import annotations

import argparse
import math

import chess
import numpy as np
import torch

from .fast_eval import network_values
from .root_ranking import terminal_value


class Node:
    __slots__ = ("moves", "q", "exact", "prior", "n", "w", "children")

    def __init__(self, moves: list[str], q: np.ndarray, exact: list[float | None],
                 tau: float) -> None:
        self.moves = moves
        self.exact = exact
        self.q = np.array([q[i] if e is None else e for i, e in enumerate(exact)])
        z = (self.q - self.q.max()) / tau
        p = np.exp(z)
        self.prior = p / p.sum()
        self.n = np.zeros(len(moves))
        self.w = np.zeros(len(moves))
        self.children: dict[int, Node] = {}

    def values(self, pessimistic: bool) -> np.ndarray:
        """What each move is worth now, for the side to move."""
        visited = self.n > 0
        mean = np.where(visited, self.w / np.maximum(self.n, 1), self.q)
        v = np.minimum(self.q, mean) if pessimistic else mean
        exact = np.array([e is not None for e in self.exact])
        return np.where(exact, self.q, v)


class MCTS:
    def __init__(self, model, centres, device: str, sims: int = 32, c_puct: float = 0.1,
                 tau: float = 0.05, pessimistic: bool = True,
                 leaf_model=None, leaf_centres=None) -> None:
        self.model, self.centres = model, centres
        self.leaf_model = model if leaf_model is None else leaf_model
        self.leaf_centres = centres if leaf_centres is None else leaf_centres
        self.device = device
        self.sims, self.c_puct, self.tau, self.pessimistic = sims, c_puct, tau, pessimistic
        self.evals = 0

    @torch.no_grad()
    def _expand(self, board: chess.Board, root: bool) -> Node:
        moves = [m.uci() for m in board.legal_moves]
        model, centres = ((self.model, self.centres) if root
                          else (self.leaf_model, self.leaf_centres))
        self.evals += 1
        q = network_values(model, centres, [board], [moves], self.device)[0]
        exact = []
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
            exact.append(terminal_value(board))
            board.pop()
        return Node(moves, q, exact, self.tau)

    def _select(self, node: Node) -> int:
        u = self.c_puct * node.prior * math.sqrt(node.n.sum() + 1) / (1 + node.n)
        return int(np.argmax(node.values(self.pessimistic) + u))

    def _simulate(self, root: Node, board: chess.Board) -> None:
        path, node, pushed = [], root, 0
        while True:
            i = self._select(node)
            path.append((node, i))
            if node.exact[i] is not None:               # the game ends: exact value
                v = node.exact[i]
                break
            board.push(chess.Move.from_uci(node.moves[i]))
            pushed += 1
            child = node.children.get(i)
            if child is None:                           # one network call, then stop
                child = self._expand(board, root=False)
                node.children[i] = child
                v = 1.0 - float(child.q.max())          # the leaf, for the mover at node
                break
            node = child
        for _ in range(pushed):
            board.pop()
        for node, i in reversed(path):                  # v is for the side choosing i
            node.n[i] += 1
            node.w[i] += v
            v = 1.0 - v

    def search(self, board: chess.Board) -> Node:
        root = self._expand(board, root=True)
        for _ in range(self.sims):
            self._simulate(root, board)
        return root

    def best_move(self, board: chess.Board) -> str:
        root = self.search(board)
        if self.sims == 0:
            return root.moves[int(root.q.argmax())]
        v = root.values(self.pessimistic)
        return root.moves[max(range(len(root.moves)), key=lambda i: (root.n[i], v[i]))]


def make_mcts_player(model, centres, device: str, sims: int, c_puct: float = 0.1,
                     tau: float = 0.05, pessimistic: bool = True, leaf_model=None,
                     leaf_centres=None, calls: list | None = None):
    """A best_move(board) callable for src/match.py and the puzzles; `calls`, if
    given, collects the network calls of every move (sims + 1)."""
    def player(board: chess.Board) -> str:
        m = MCTS(model, centres, device, sims, c_puct, tau, pessimistic,
                 leaf_model, leaf_centres)
        move = m.best_move(board)
        if calls is not None:
            calls.append(m.evals)
        return move
    return player


def puzzles(pick, puzzles_csv: str, num: int, start: int = 0,
            deadline: float | None = None) -> dict:
    """Puzzle accuracy of `pick` on puzzles start..start+num of the file, with one 0/1
    per puzzle so two settings can be paired. Past `deadline` (a time.time()) it stops
    between puzzles: the rows are then the first `attempted` puzzles, still pairable."""
    import csv
    import time
    from .puzzles_eval import solves_line
    rows = []
    stopped = False
    with open(puzzles_csv, newline="") as fh:
        for k, row in enumerate(csv.DictReader(fh)):
            if k < start:
                continue
            if len(rows) >= num:
                break
            if deadline is not None and time.time() > deadline:
                stopped = True
                break
            rows.append(int(solves_line(chess.Board(row["FEN"]), row["Moves"].split(), pick)))
    return {"solved": sum(rows), "attempted": len(rows),
            "accuracy": sum(rows) / max(1, len(rows)), "stopped_early": stopped, "rows": rows}


if __name__ == "__main__":
    import json
    import time
    from .puzzles_eval import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--leaf_ckpt", default=None)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--c_puct", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--no_pessimism", action="store_true")
    ap.add_argument("--depth2", action="store_true",
                    help="the bot's alpha-beta at depth 2 instead, as the reference")
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hours", type=float, default=None,
                    help="stop between puzzles after this long and write what was done")
    a = ap.parse_args()
    model, centres, _ = load_model(a.ckpt, a.device)
    leaf = load_model(a.leaf_ckpt, a.device)[:2] if a.leaf_ckpt else (None, None)
    calls: list[int] = []
    if a.depth2:
        from .root_ranking import make_onepass_player
        two = make_onepass_player((model, centres), leaf if a.leaf_ckpt else None, a.device, 2)
        pick = lambda board: calls.append(2) or two(board)      # noqa: E731
    else:
        pick = make_mcts_player(model, centres, a.device, a.sims, a.c_puct, a.tau,
                                not a.no_pessimism, *leaf, calls=calls)
    t = time.time()
    res = puzzles(pick, a.puzzles, a.num, a.start,
                  None if a.hours is None else t + a.hours * 3600)
    res.update({"player": "depth2" if a.depth2 else "mcts", "sims": a.sims, "c_puct": a.c_puct, "tau": a.tau,
                "pessimistic": not a.no_pessimism, "ckpt": a.ckpt, "leaf_ckpt": a.leaf_ckpt,
                "seconds": round(time.time() - t), "moves": len(calls),
                "seconds_per_move": (time.time() - t) / max(1, len(calls))})
    print(f"[mcts] {res['player']} sims {a.sims} c {a.c_puct} tau {a.tau}: {res['solved']}/{res['attempted']}"
          f" = {res['accuracy'] * 100:.1f}%, {res['seconds_per_move']:.3f} s/move", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res) + "\n")
