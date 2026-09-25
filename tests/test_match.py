"""The match harness: Elo arithmetic, colour pairing, and how a game ends."""

import chess
import numpy as np
import pytest

from src import match
from src.match import elo_from_score, play, run_match, summarise


def test_elo_of_known_scores():
    assert elo_from_score(0.5) == 0.0
    assert elo_from_score(0.76) == pytest.approx(200.2, abs=0.1)
    assert elo_from_score(0.24) == pytest.approx(-200.2, abs=0.1)


def test_summary_counts_and_an_interval_around_the_score():
    s = summarise([1, 1, 0.5, 0, 1, 0.5, 0, 1])
    assert (s["games"], s["wins"], s["draws"], s["losses"]) == (8, 4, 2, 2)
    assert s["score"] == pytest.approx(5 / 8)
    assert s["elo_low"] < s["elo"] < s["elo_high"]


def test_colours_are_swapped_and_scored_from_a_s_side(monkeypatch):
    """If every game were won by white, a fair pairing must score it exactly even.
    Forgetting to swap colours, or to invert the second result, breaks this."""
    calls = []

    def white_always_wins(fen, white, black, max_plies=300):
        calls.append((fen, white, black))
        return 1.0

    monkeypatch.setattr(match, "play", white_always_wins)
    results = run_match(["p1", "p2"], "A", "B", log=lambda _: None)
    assert calls == [("p1", "A", "B"), ("p1", "B", "A"), ("p2", "A", "B"), ("p2", "B", "A")]
    assert results == [1.0, 0.0, 1.0, 0.0]
    assert summarise(results)["elo"] == 0.0


def first_legal(board):
    return next(iter(board.legal_moves)).uci()


def mate_if_possible(board):
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move.uci()
    return first_legal(board)


def test_a_game_ends_on_mate_and_reports_the_winner():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"               # Ra8#
    assert play(fen, mate_if_possible, first_legal) == 1.0
    black_to_mate = "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"     # ...Ra1#
    assert play(black_to_mate, first_legal, mate_if_possible) == 0.0


def test_a_game_that_does_not_finish_is_a_draw():
    assert play(chess.STARTING_FEN, first_legal, first_legal, max_plies=4) == 0.5


def test_starting_positions_are_balanced_middlegames(monkeypatch):
    from src import agreement_eval
    middlegame = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9"
    items = [
        (middlegame, [], np.array([0.52, 0.40])),                  # kept
        (middlegame.replace("0 9", "0 10"), [], np.array([0.91, 0.30])),  # winning: out
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [],
         np.array([0.52, 0.50])),                                   # opening: out
    ]
    monkeypatch.setattr(agreement_eval, "complete_positions", lambda bag: items)
    assert match.balanced_positions("unused", 1) == [middlegame]
    with pytest.raises(ValueError):
        match.balanced_positions("unused", 2)


def test_first_to_stops_as_soon_as_one_side_gets_there(monkeypatch):
    """White always wins, so A wins every odd game and loses every even one:
    1-0, 1-1, 2-1, 2-2, 3-2 -- first to 3 is decided at game 5, not later."""
    monkeypatch.setattr(match, "play", lambda fen, w, b, max_plies=300: 1.0)
    results = run_match([f"p{i}" for i in range(10)], "A", "B", log=lambda _: None,
                        first_to=3)
    assert results == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert match.decided(results, 3) == "A"


def test_level_at_the_target_is_not_a_win(monkeypatch):
    """All draws: both reach the target together, nobody leads, the match runs on
    until the positions run out, and nobody is declared the winner."""
    monkeypatch.setattr(match, "play", lambda fen, w, b, max_plies=300: 0.5)
    results = run_match(["p1", "p2", "p3"], "A", "B", log=lambda _: None, first_to=2)
    assert len(results) == 6
    assert match.decided(results, 2) is None


def test_a_long_match_saves_after_every_position_and_stops_at_its_deadline(monkeypatch):
    """A cut must never split a position's two colours, and what was played is kept."""
    import time as clock
    monkeypatch.setattr(match, "play", lambda fen, w, b, max_plies=300: 1.0)
    saved = []
    results = run_match(["p1", "p2", "p3"], "A", "B", log=lambda _: None,
                        on_position=lambda r: saved.append(len(r)))
    assert saved == [2, 4, 6]

    results = run_match(["p1", "p2"], "A", "B", log=lambda _: None,
                        deadline=clock.time() - 1)       # already past: play nothing
    assert results == []


def test_an_empty_match_summarises_without_nan():
    s = summarise([])
    assert s["games"] == 0 and s["elo"] is None
