"""Alpha-beta search using the network as its evaluation function.

    python -m src.search --ckpt checkpoints/chess_best.pt --depth 2

The measurements said where the strength is going: the model's median move gives
up 3.5 points of win probability, but 17.5% of its moves give up more than 15 --
a blunder every six moves. Most one-ply blunders are visible from two ply, so a
shallow search should convert directly into strength.

The network is unusually good at guiding the search. It returns Q(s,a) for every
legal move in a single batch, which is both the leaf evaluation (V(s) = max_a
Q(s,a)) and the move ordering -- and alpha-beta's pruning lives or dies on move
ordering. One network call per node, not one per move.

Values are win probabilities in [0, 1] from the side to move, so the negamax flip
is 1 - v rather than -v.

A move's value is the MINIMUM of the network's direct Q(s,a) and the value backed
up from the child. Both estimate the same quantity -- the position after the move
-- and taking the worse of the two is what keeps search from being fooled.

Why that matters here, measured: after 1.e4 d5 2.Nc3 dxe4 3.Nxe4, the network
rates 3...Qxd2+ at 0.05, correctly seeing that it drops the queen. But in the
position after Qxd2+, it rates White's queen-winning captures at only 0.25-0.35,
because a position where a queen hangs after a pointless check is nothing like the
Lichess games it trained on -- it is out of distribution, and the network answers
near the mean. Backing that up gives 1 - 0.35 = 0.65 and search happily plays the
blunder.

Alpha-beta is a maximiser, so it does not merely stumble into these positions: it
hunts for wherever the evaluator is most wrong. Deeper search amplifies
out-of-distribution error rather than averaging it out. Taking the min is the
cheap pessimistic fix (the same instinct as conservative value estimates in
offline RL): a move has to look good BOTH before and after it is played.
"""

from __future__ import annotations

import argparse
import time

import chess
import numpy as np
import torch

from .chess_data import ACTION_OFFSET, MOVE_TO_ACTION, tokenize_fen
from .fast_eval import network_values
from .puzzles_eval import load_model


class Searcher:
    def __init__(self, model, centres, device: str, shared: bool = False,
                 leaf_model=None, leaf_centres=None) -> None:
        """`shared` evaluates the moves of a node with the position's prefix computed
        once (src/fast_eval.py): exact to float rounding and ~6x faster on CPU.

        `leaf_model` scores every node below the root, `model` only the root: the
        root's Q(s, a) and the backed-up values can come from different networks.
        That is the test of whether a fine-tune changed its root ranking or its
        values of the positions after a move. Without it both are `model`.
        """
        self.model = model
        self.centres = centres
        self.device = device
        self.shared = shared
        self.leaf_model = model if leaf_model is None else leaf_model
        self.leaf_centres = centres if leaf_centres is None else leaf_centres
        self.nodes = 0
        self.evals = 0

    @torch.no_grad()
    def _q_values(self, board: chess.Board, moves: list[str], root: bool = False) -> np.ndarray:
        """Q(s,a) for every legal move, one batch, terminal states scored exactly."""
        self.evals += 1
        model, centres = (self.model, self.centres) if root else (self.leaf_model, self.leaf_centres)
        if self.shared:
            values = network_values(model, centres, [board], [moves], self.device)[0]
        else:
            state = torch.from_numpy(tokenize_fen(board.fen()).astype(np.int64))
            actions = torch.tensor([MOVE_TO_ACTION[m] + ACTION_OFFSET for m in moves], dtype=torch.long)
            x = torch.cat([state.repeat(len(moves), 1), actions[:, None]], dim=1).to(self.device)
            logits, _ = model(x)
            values = (logits.softmax(-1) * centres).sum(-1).cpu().numpy().astype(np.float64)

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
        return values

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Value of `board` for the side to move, in [0, 1]."""
        self.nodes += 1
        moves = [m.uci() for m in board.legal_moves]
        if not moves:                                   # checkmate or stalemate
            return 0.0 if board.is_check() else 0.5

        q = self._q_values(board, moves)
        order = np.argsort(q)[::-1]                     # network-guided ordering
        if depth <= 1:
            return float(q[order[0]])

        best = 0.0
        for i in order:
            uci = moves[int(i)]
            if q[i] in (0.5, 1.0):                      # exact terminal, no recursion
                value = q[i]
            else:
                board.push(chess.Move.from_uci(uci))
                backed_up = 1.0 - self._search(board, depth - 1, 1.0 - beta, 1.0 - alpha)
                board.pop()
                value = min(float(q[i]), backed_up)     # see _combine below
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:                           # the opponent avoids this line
                break
        return best

    def best_move(self, board: chess.Board, depth: int = 2) -> str:
        moves = [m.uci() for m in board.legal_moves]
        q = self._q_values(board, moves, root=True)
        if depth <= 1:
            return moves[int(q.argmax())]

        best_uci, best_value, alpha = moves[int(q.argmax())], -1.0, 0.0
        for i in np.argsort(q)[::-1]:
            uci = moves[int(i)]
            if q[i] in (0.5, 1.0):
                value = float(q[i])
            else:
                board.push(chess.Move.from_uci(uci))
                backed_up = 1.0 - self._search(board, depth - 1, 0.0, 1.0 - alpha)
                board.pop()
                value = min(float(q[i]), backed_up)     # see _combine below
            if value > best_value:
                best_value, best_uci = value, uci
            alpha = max(alpha, best_value)
        return best_uci


def make_player(model, centres, device: str, depth: int, shared: bool = False,
                leaf_model=None, leaf_centres=None, nodes: list | None = None):
    """A best_move(board) callable at a fixed depth, for play_eval and the bot.
    `nodes`, if given, collects the network calls of every move."""
    def player(board: chess.Board) -> str:
        searcher = Searcher(model, centres, device, shared, leaf_model, leaf_centres)
        move = searcher.best_move(board, depth)
        if nodes is not None:
            nodes.append(searcher.evals)
        return move
    return player


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--shared", action="store_true", help="share the position prefix (exact, faster)")
    args = ap.parse_args()

    model, centres, _ = load_model(args.ckpt, args.device)
    board = chess.Board()
    total_nodes = total_time = 0.0
    for ply in range(args.positions):
        if board.is_game_over():
            break
        searcher = Searcher(model, centres, args.device, args.shared)
        t = time.time()
        uci = searcher.best_move(board, args.depth)
        dt = time.time() - t
        total_nodes += searcher.nodes
        total_time += dt
        print(f"  ply {ply:2d}  {uci}  {searcher.nodes:5d} nodes  "
              f"{searcher.evals:4d} net calls  {dt:6.2f}s")
        board.push(chess.Move.from_uci(uci))
    print(f"[search] depth {args.depth}: {total_nodes/args.positions:.0f} nodes/move, "
          f"{total_time/args.positions:.2f}s/move")
