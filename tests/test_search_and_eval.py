"""The two functions every reported number goes through: move choice and search.

Until these tests existed, the suite stayed green with best_move picking the WORST
move, with the negamax flip removed, and with the pessimistic min() deleted -- each
checked by mutating a copy of the code, and each mutation is now caught. The results were right anyway (a model that
picked the worst move would not solve 60% of puzzles, and a broken search would not
win a match 10-0), but nothing would have noticed a regression.

The network is replaced by stubs so that what is tested is the logic, not a model:
a material count stands in for Q(s, a), and specific positions can be made to lie.
"""

import chess
import numpy as np
import pytest
import torch

from src.chess_data import ACTION_OFFSET, MOVE_TO_ACTION
from src.puzzles_eval import best_move
from src.search import Searcher

PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board: chess.Board, colour: bool) -> int:
    return sum(PIECE.get(p.piece_type, 0) * (1 if p.color == colour else -1)
               for p in board.piece_map().values())


def material_q(board: chess.Board, moves: list[str]) -> np.ndarray:
    """Q(s, a) for the side to move: its material balance after the move, squashed."""
    me, out = board.turn, []
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        out.append(np.clip(0.5 + 0.04 * material(board, me), 0.01, 0.99))
        board.pop()
    return np.array(out, dtype=np.float64)


def searcher_with(q_fn) -> Searcher:
    s = Searcher(model=None, centres=None, device="cpu")
    s._q_values = q_fn
    return s


# White can win a pawn with the queen, but the pawn is defended: Qxd5 exd5.
DEFENDED_PAWN = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"


def test_the_premise_of_the_defended_pawn_position():
    board = chess.Board(DEFENDED_PAWN)
    moves = [m.uci() for m in board.legal_moves]
    q = material_q(board, moves)
    assert moves[int(q.argmax())] == "d1d5"           # one ply deep, it looks like a win
    board.push(chess.Move.from_uci("d1d5"))
    assert chess.Move.from_uci("e6d5") in board.legal_moves   # and it is not


@pytest.mark.parametrize("depth", [2, 3])
def test_search_does_not_trade_the_queen_for_a_pawn(depth):
    """Kills a wrong flip at the root: without `1 - v`, the opponent's good reply is
    read as good for us and the search walks into the recapture. It does NOT catch
    the same bug inside the recursion -- verified by mutation: near 0.5, v and 1 - v
    are close and the choice survives. test_alpha_beta_returns_the_exact_negamax_value
    is the test for that."""
    move = searcher_with(material_q).best_move(chess.Board(DEFENDED_PAWN), depth)
    assert move != "d1d5"


def _overconfident_after(trap: str):
    """A network that is badly wrong in one place: after `trap`, every reply of the
    opponent looks hopeless to the opponent. That is what an out-of-distribution
    position does, and it is what a maximiser goes looking for."""
    start = chess.Board()
    start.push(chess.Move.from_uci(trap))
    lying = start.board_fen()

    def q(board, moves):
        if board.board_fen() == lying:
            return np.full(len(moves), 0.05)
        if board.board_fen() == chess.Board().board_fen():
            return np.array([0.30 if m == trap else 0.60 if m == "e2e4" else 0.40
                             for m in moves])
        return np.full(len(moves), 0.40)
    return q


def test_search_does_not_trust_a_child_over_the_move_itself():
    """Kills removal of the pessimistic min() at the root: the trap's direct estimate
    is 0.30, its backed-up value a delusional 0.95. The move is worth the lower."""
    move = searcher_with(_overconfident_after("a2a3")).best_move(chess.Board(), 2)
    assert move == "e2e4"


def test_a_node_is_never_worth_more_than_its_best_direct_estimate():
    """The same invariant inside the recursion: with min(), V(s) <= max Q(s, a)."""
    s = searcher_with(_overconfident_after("a2a3"))
    board = chess.Board()
    q_root = s._q_values(board, [m.uci() for m in board.legal_moves])
    assert s._search(board, 2, 0.0, 1.0) <= q_root.max() + 1e-12


def test_search_scores_a_mate_as_a_win():
    """A mate is terminal and exact; nothing the network says can outvote it."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")       # Ra8#
    s = Searcher(model=_stub_model(favourite=None), centres=_CENTRES, device="cpu")
    assert s.best_move(board, 2) == "a1a8"


# --- best_move, the function behind every puzzle number ---------------------

_CENTRES = torch.tensor((np.arange(4) + 0.5) / 4, dtype=torch.float32)


def _stub_model(favourite: str | None):
    """A 'network' that puts all its mass on the top bucket for one move and on the
    bottom bucket for every other move. It reads the action token, as the real head
    does, so it knows which move each row of the batch is."""
    target = None if favourite is None else MOVE_TO_ACTION[favourite] + ACTION_OFFSET

    def model(x):
        logits = torch.full((len(x), 4), -20.0)
        logits[:, 0] = 20.0                              # bottom bucket by default
        if target is not None:
            hit = x[:, -1] == target
            logits[hit, 0], logits[hit, 3] = -20.0, 20.0
        return logits, None
    return model


@pytest.mark.parametrize("favourite", ["g1f3", "e2e4", "b1a3"])
def test_best_move_picks_the_highest_expected_value(favourite):
    board = chess.Board()
    assert best_move(_stub_model(favourite), _CENTRES, board, "cpu") == favourite


def test_best_move_leaves_the_board_as_it_found_it():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    before = board.fen()
    best_move(_stub_model(None), _CENTRES, board, "cpu")
    assert board.fen() == before


# --- the exact value, against a reference with no pruning -------------------

def _q_off_terminal(board, moves):
    """material_q nudged off 0.5 and 1.0: the searcher treats those two values as
    exact terminal scores and stops recursing, which is right for the real network
    (its softmax never lands on them exactly) but would make a stub look terminal."""
    return material_q(board, moves) + 1e-3


def reference_value(board: chess.Board, depth: int) -> float:
    """Plain negamax with the same pessimistic combination, no alpha-beta, no
    ordering: slow, and too simple to share a bug with the real one."""
    moves = [m.uci() for m in board.legal_moves]
    if not moves:
        return 0.0 if board.is_check() else 0.5
    q = _q_off_terminal(board, moves)
    if depth <= 1:
        return float(q.max())
    best = 0.0
    for uci, qi in zip(moves, q):
        board.push(chess.Move.from_uci(uci))
        child = 1.0 - reference_value(board, depth - 1)
        board.pop()
        best = max(best, min(float(qi), child))
    return best


MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


@pytest.mark.parametrize("fen", [DEFENDED_PAWN, MIDDLEGAME])
@pytest.mark.parametrize("depth", [2, 3])
def test_alpha_beta_returns_the_exact_negamax_value(fen, depth):
    """Pruning must not change the value, only the work. Catches a wrong flip inside
    the recursion, which the move-choice tests above cannot: near 0.5, v and 1 - v
    are close enough that the chosen move often survives the bug."""
    s = searcher_with(_q_off_terminal)
    board = chess.Board(fen)
    assert s._search(board, depth, 0.0, 1.0) == pytest.approx(reference_value(board, depth))
    assert board.fen() == chess.Board(fen).fen()       # the board is restored


@pytest.mark.parametrize("fen", [DEFENDED_PAWN, MIDDLEGAME])
@pytest.mark.parametrize("depth", [2, 3])
def test_the_chosen_move_achieves_the_best_value(fen, depth):
    """The root is separate code (best_move), so it gets the same check: the move it
    returns must be worth the reference value of the whole position."""
    s = searcher_with(_q_off_terminal)
    board = chess.Board(fen)
    move = s.best_move(board, depth)
    moves = [m.uci() for m in board.legal_moves]
    q = dict(zip(moves, _q_off_terminal(board, moves)))
    board.push(chess.Move.from_uci(move))
    worth = min(float(q[move]), 1.0 - reference_value(board, depth - 1))
    board.pop()
    assert worth == pytest.approx(reference_value(board, depth))
