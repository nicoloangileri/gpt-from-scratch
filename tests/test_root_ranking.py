"""The one-pass shortcut must be the search, decision for decision.

src/root_ranking.py claims that at depth 2 the root's choice is argmax min(Q_root,
backed-up), so every root/leaf pairing can be read off one evaluation of the root and
its children. These tests hold it to src/search.py (plain forward, recursion and all)
on random positions and on positions where a move ends the game, and to
puzzles_eval.best_move at depth 1.
"""
import chess
import numpy as np
import pytest

from src.fast_eval import sample_positions
from src.puzzles_eval import best_move, solves_line
from src.root_ranking import (choose, configurations, depth2_choice, diagnose, evaluate_node,
                              solve_all, terminal_value)
from src.search import Searcher
from tests.test_fast_eval import CENTRES, tiny_model

A, B = tiny_model(10), tiny_model(11)
MODELS = {"a": (A, CENTRES), "b": (B, CENTRES)}

POSITIONS = [b.fen() for b in sample_positions(10, seed=7)] + [
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",          # Ra8 mates
    "7k/8/5Q2/8/8/8/8/6K1 w - - 0 1",             # Qf7 stalemates, Qf8 mates
    "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1",         # a defended pawn
]


@pytest.mark.parametrize("fen", POSITIONS)
def test_depth2_is_the_searcher(fen):
    node = evaluate_node(MODELS, chess.Board(fen), "cpu")
    for root, leaf in (("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")):
        s = Searcher(MODELS[root][0], CENTRES, "cpu", leaf_model=MODELS[leaf][0])
        assert choose(node, root, leaf, 2) == s.best_move(chess.Board(fen), 2), (root, leaf)


@pytest.mark.parametrize("fen", POSITIONS)
def test_depth1_is_best_move(fen):
    node = evaluate_node(MODELS, chess.Board(fen), "cpu")
    for name, (model, _) in MODELS.items():
        assert choose(node, name, name, 1) == best_move(model, CENTRES, chess.Board(fen), "cpu")


def test_the_hybrid_really_mixes_the_networks():
    """If the leaf network were ignored, the two hybrids would be the plain configs;
    on enough positions some choice must differ."""
    differs = 0
    for fen in POSITIONS:
        node = evaluate_node(MODELS, chess.Board(fen), "cpu")
        differs += choose(node, "a", "b", 2) != choose(node, "a", "a", 2)
    assert differs > 0


def test_terminal_values():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    board.push_uci("a1a8")
    assert terminal_value(board) == 1.0
    board = chess.Board("7k/8/5Q2/8/8/8/8/6K1 w - - 0 1")
    board.push_uci("f6f7")
    assert terminal_value(board) == 0.5
    assert terminal_value(chess.Board()) is None


def test_ties_break_like_the_searcher():
    """Two moves worth the same at depth 2: the one the root ranks higher wins."""
    q = np.array([0.6, 0.9, 0.7])
    bu = np.array([0.6, 0.5, 0.5])          # values 0.6, 0.5, 0.5
    assert depth2_choice(q, bu) == 0
    bu = np.array([0.4, 0.5, 0.5])          # values 0.4, 0.5, 0.5: 0.9 beats 0.7
    assert depth2_choice(q, bu) == 1


def test_solve_all_is_solves_line():
    """Every configuration of the one pass gets the 0/1 that solves_line would."""
    configs = configurations(["a", "b"])
    lines = []
    for board in sample_positions(6, seed=8):
        rng = np.random.default_rng(len(lines))
        b, line = board.copy(), []
        for _ in range(4):
            moves = list(b.legal_moves)
            if not moves:
                break
            m = moves[int(rng.integers(len(moves)))]
            line.append(m.uci())
            b.push(m)
        lines.append((board.fen(), line))
    # and a real puzzle shape: the opponent's move, then a mate to find
    lines.append(("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", ["g8h8", "a1a8"]))
    for fen, line in lines:
        got, _ = solve_all(MODELS, fen, line, "cpu", configs)
        for key, (root, leaf, depth) in configs.items():
            s = Searcher(MODELS[root][0], CENTRES, "cpu", leaf_model=MODELS[leaf][0])
            pick = ((lambda bd, s=s: s.best_move(bd, 2)) if depth == 2 else
                    (lambda bd, m=MODELS[root][0]: best_move(m, CENTRES, bd, "cpu")))
            assert got[key] == int(solves_line(chess.Board(fen), line, pick)), (fen, line, key)


def test_diagnose_ranks_the_solution():
    node = {"moves": ["x", "y", "z"],
            "q": {"a": np.array([0.2, 0.9, 0.5])},
            "bu": {"a": np.array([0.8, 0.1, 0.5])}}
    d = diagnose(node, "y")
    assert d["q_rank"]["a"] == 1 and d["bu_rank"]["a"] == 3
    assert d["d2_rank"]["a/a"] == 3                    # min: 0.2, 0.1, 0.5


def test_a_different_mate_also_solves(monkeypatch):
    """As on Lichess: Rb8 mates as well as the listed Ra8, and a non-mate does not."""
    import src.root_ranking as rr
    fen, line = "6k1/5ppp/8/8/8/8/8/RR4K1 b - - 0 1", ["g8h8", "a1a8"]
    picks = {"a@1": "b1b8", "a@2": "a1a2"}
    monkeypatch.setattr(rr, "choose", lambda node, root, leaf, depth: picks[f"{root}@{depth}"])
    got, _ = rr.solve_all(MODELS, fen, line, "cpu", {"a@1": ("a", "a", 1), "a@2": ("a", "a", 2)})
    assert got == {"a@1": 1, "a@2": 0}


@pytest.mark.parametrize("fen", POSITIONS)
def test_the_onepass_player_is_the_searcher(fen):
    """src/match.py --onepass plays these moves; they must be the search's, for the
    hybrid (root a, leaves b) as for a single network, at depth 1 and 2."""
    from src.root_ranking import make_onepass_player
    for leaf in (None, MODELS["b"]):
        player = make_onepass_player(MODELS["a"], leaf, "cpu", depth=2)
        s = Searcher(A, CENTRES, "cpu", leaf_model=None if leaf is None else B)
        assert player(chess.Board(fen)) == s.best_move(chess.Board(fen), 2)
    d1 = make_onepass_player(MODELS["a"], None, "cpu", depth=1)
    assert d1(chess.Board(fen)) == best_move(A, CENTRES, chess.Board(fen), "cpu")


def test_the_onepass_player_refuses_depth_3():
    from src.root_ranking import make_onepass_player
    with pytest.raises(ValueError):
        make_onepass_player(MODELS["a"], None, "cpu", depth=3)


def test_match_wires_the_leaves_and_the_onepass_player(monkeypatch, tmp_path):
    """The command line the Kaggle session runs: A searches with B's leaves. Each
    player built by main() must make the move the matching Searcher makes."""
    import json
    import sys

    from src import match, puzzles_eval
    fen = POSITIONS[0]
    nets = {"a.pt": A, "b.pt": B}
    monkeypatch.setattr(puzzles_eval, "load_model", lambda p, d: (nets[p], CENTRES, {}))
    monkeypatch.setattr(match, "balanced_positions", lambda bag, n, seed: [fen])
    seen = {}

    def one_move_each(start, white, black, max_plies=300):
        seen.setdefault("white", []).append(white(chess.Board(start)))
        seen.setdefault("black", []).append(black(chess.Board(start)))
        return 0.5
    monkeypatch.setattr(match, "play", one_move_each)
    out = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", ["match", "--a", "a.pt", "--a_leaf", "b.pt", "--b", "b.pt",
                                      "--depth", "2", "--positions", "1", "--onepass",
                                      "--threads", "1", "--out", str(out)])
    match.main()
    hybrid = Searcher(A, CENTRES, "cpu", leaf_model=B).best_move(chess.Board(fen), 2)
    plain_b = Searcher(B, CENTRES, "cpu").best_move(chess.Board(fen), 2)
    assert seen["white"][0] == hybrid and seen["black"][1] == hybrid     # A, both colours
    assert seen["black"][0] == plain_b and seen["white"][1] == plain_b
    summary = json.loads(out.read_text())
    assert summary["a_leaf"] == "b.pt" and summary["onepass"] and summary["games"] == 2


def test_match_gives_each_side_its_depth_and_times_it(monkeypatch, tmp_path):
    """The depth match: A at 3, B at 2, the same network. A's moves must be the
    depth-3 search's, B's the depth-2 search's, and the slice must pick its share."""
    import json
    import sys

    from src import match, puzzles_eval
    monkeypatch.setattr(puzzles_eval, "load_model", lambda p, d: (A, CENTRES, {}))
    monkeypatch.setattr(match, "balanced_positions", lambda bag, n, seed: POSITIONS[:n])
    seen = []

    def one_move_each(start, white, black, max_plies=300):
        seen.append((start, white(chess.Board(start)), black(chess.Board(start))))
        return 1.0
    monkeypatch.setattr(match, "play", one_move_each)
    out = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", ["match", "--a", "a.pt", "--b", "a.pt", "--a_depth", "3",
                                      "--b_depth", "2", "--positions", "4", "--slice", "1/2",
                                      "--onepass", "--threads", "1", "--out", str(out)])
    match.main()
    assert [s[0] for s in seen[::2]] == POSITIONS[1:4:2]        # the second and the fourth
    d3 = lambda fen: Searcher(A, CENTRES, "cpu").best_move(chess.Board(fen), 3)
    d2 = lambda fen: Searcher(A, CENTRES, "cpu").best_move(chess.Board(fen), 2)
    assert seen[0][1] == d3(seen[0][0]) and seen[0][2] == d2(seen[0][0])  # A is white
    assert seen[1][2] == d3(seen[1][0]) and seen[1][1] == d2(seen[1][0])  # A is black
    summary = json.loads(out.read_text())
    assert (summary["a_depth"], summary["b_depth"], summary["slice"]) == (3, 2, "1/2")
    assert summary["a_time"]["moves"] == 4 and summary["b_time"]["moves"] == 4
    assert summary["a_time"]["network_calls_mean"] > 2      # depth 2 here makes two calls
    assert summary["results"] == [1.0, 0.0, 1.0, 0.0]       # A's side of white's wins
