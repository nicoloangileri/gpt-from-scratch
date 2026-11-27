"""The network-guided tree search: exact where the game ends, sane where it does not."""
import chess
import numpy as np
import pytest

from src.fast_eval import sample_positions
from src.mcts import MCTS, Node, make_mcts_player
from src.puzzles_eval import best_move
from tests.test_fast_eval import CENTRES, tiny_model

A, B = tiny_model(10), tiny_model(11)
POSITIONS = [b.fen() for b in sample_positions(6, seed=7)]


@pytest.mark.parametrize("fen", POSITIONS)
def test_no_simulations_is_the_bare_network(fen):
    m = MCTS(A, CENTRES, "cpu", sims=0)
    assert m.best_move(chess.Board(fen)) == best_move(A, CENTRES, chess.Board(fen), "cpu")


@pytest.mark.parametrize("fen", POSITIONS)
def test_visits_add_up_and_the_board_is_left_as_found(fen):
    board = chess.Board(fen)
    m = MCTS(A, CENTRES, "cpu", sims=24, c_puct=0.5, leaf_model=B)
    root = m.search(board)
    assert board.fen() == fen
    assert root.n.sum() == 24
    assert m.evals == 1 + sum(1 for _ in _nodes(root)) - 1     # one call per node
    for node in _nodes(root):
        assert (node.w <= node.n + 1e-9).all() and (node.w >= -1e-9).all()


def _nodes(node):
    yield node
    for c in node.children.values():
        yield from _nodes(c)


def test_a_mate_in_one_is_played():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    for sims in (1, 8, 32):
        assert MCTS(A, CENTRES, "cpu", sims=sims).best_move(board) == "a1a8"


def test_the_mate_is_found_even_where_the_network_hates_it():
    """A random network rates the mating move anywhere; its exact value must win."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    root = MCTS(B, CENTRES, "cpu", sims=16).search(board)
    i = root.moves.index("a1a8")
    assert root.q[i] == 1.0 and root.values(True)[i] == 1.0


def test_pessimism_takes_the_worse_of_the_two_estimates():
    node = Node(["a", "b"], np.array([0.7, 0.4]), [None, None], tau=0.05)
    node.n[:] = [2, 2]
    node.w[:] = [1.8, 0.4]                 # backed-up means 0.9 and 0.2
    assert np.allclose(node.values(True), [0.7, 0.2])
    assert np.allclose(node.values(False), [0.9, 0.2])


def test_the_player_counts_its_network_calls():
    calls = []
    pick = make_mcts_player(A, CENTRES, "cpu", sims=10, calls=calls)
    pick(chess.Board(POSITIONS[0]))
    assert len(calls) == 1 and 2 <= calls[0] <= 11


def test_a_deadline_stops_between_puzzles_and_keeps_the_first_ones(tmp_path):
    """A job cut by the session's deadline must leave the first puzzles, pairable."""
    import csv
    import time
    from src.mcts import puzzles
    f = tmp_path / "p.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["PuzzleId", "FEN", "Moves"])
        for i in range(5):
            w.writerow([i, chess.STARTING_FEN, "e2e4 e7e5"])
    picked = []
    pick = lambda board: picked.append(1) or "e7e5"          # noqa: E731
    full = puzzles(pick, str(f), 5)
    assert full["attempted"] == 5 and not full["stopped_early"]
    cut = puzzles(pick, str(f), 5, deadline=time.time() - 1)
    assert cut["attempted"] == 0 and cut["stopped_early"] and cut["rows"] == []
