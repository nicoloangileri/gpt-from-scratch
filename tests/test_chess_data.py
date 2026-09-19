"""Tests for the ChessBench ingestion path."""

import struct

import chess
import numpy as np
import pytest

from src.chess_data import (
    ACTION_OFFSET, FULL_VOCAB, MOVE_TO_ACTION, NUM_ACTIONS, SEQUENCE_LENGTH,
    iter_action_value, tokenize_fen, win_prob_to_bucket,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_move_vocabulary_matches_paper():
    assert NUM_ACTIONS == 1968
    assert FULL_VOCAB == ACTION_OFFSET + NUM_ACTIONS


def test_every_legal_move_of_a_real_game_is_in_the_vocabulary():
    board = chess.Board()
    for _ in range(40):
        moves = list(board.legal_moves)
        if not moves:
            break
        assert all(m.uci() in MOVE_TO_ACTION for m in moves)
        board.push(moves[0])


def test_tokenize_fen_is_fixed_length():
    for fen in (START_FEN,
                "8/8/1r6/1Bk5/1RPN4/8/1K6/8 w - - 3 69",           # no castling, no ep
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"):  # ep square
        tokens = tokenize_fen(fen)
        assert tokens.shape == (SEQUENCE_LENGTH,)
        assert tokens.dtype == np.uint8
        assert tokens.max() < ACTION_OFFSET


def test_tokenize_fen_distinguishes_side_to_move():
    white = tokenize_fen(START_FEN)
    black = tokenize_fen(START_FEN.replace(" w ", " b "))
    assert not np.array_equal(white, black)


def test_win_prob_buckets_are_monotone_and_in_range():
    probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    buckets = win_prob_to_bucket(probs, 64)
    assert list(buckets) == sorted(buckets)
    assert buckets[0] == 0 and buckets[-1] == 63


def test_bag_reader_round_trip(tmp_path):
    """Write a bag by hand (records + int64 limits + index offset) and read it back."""
    records = [("8/8/1r6/1Bk5/1RPN4/8/1K6/8 w - - 3 69", "b5c6", 0.4945),
               (START_FEN, "e2e4", 0.5321)]
    blob, limits = b"", []
    for fen, move, win_prob in records:
        blob += (bytes([len(fen)]) + fen.encode()
                 + bytes([len(move)]) + move.encode()
                 + struct.pack(">d", win_prob))
        limits.append(len(blob))
    path = tmp_path / "sample.bag"
    path.write_bytes(blob + b"".join(struct.pack("<q", l) for l in limits)
                     + struct.pack("<Q", len(blob)))

    got = list(iter_action_value(path))
    assert len(got) == len(records)
    for (fen, move, win_prob), (g_fen, g_move, g_win) in zip(records, got):
        assert (fen, move) == (g_fen, g_move)
        assert g_win == pytest.approx(win_prob)


def test_bag_reader_honours_limit(tmp_path):
    path = tmp_path / "one.bag"
    fen, move = START_FEN, "e2e4"
    blob = bytes([len(fen)]) + fen.encode() + bytes([len(move)]) + move.encode() + struct.pack(">d", 0.5)
    path.write_bytes(blob + struct.pack("<q", len(blob)) + struct.pack("<Q", len(blob)))
    assert len(list(iter_action_value(path, limit=1))) == 1


def test_real_mate_puzzle_is_delivered_by_the_solver():
    """Semantic check of the puzzle format, on real data (lichess_db_puzzle 000Zo).

    Replaces a test that compared Moves[0] with the first SAN of ChessBench's
    Solution column -- a tautology, since that column is just the whole Moves line in
    SAN, and it had certified the wrong parity. Here the check cannot agree with
    itself: in a mate puzzle the side that ends up mated is the opponent, so if the
    side to move in the FEN is the one mated, the FEN precedes the opponent's move
    and the solver plays the odd plies.
    """
    fen = "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35"
    board = chess.Board(fen)
    first_mover = board.turn
    for uci in "e5f6 e8e1 g1f2 e1f1".split():
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    assert board.turn == first_mover        # the side to move when mated = the loser


def test_solves_line_plays_the_opponent_first_and_checks_the_solver():
    from src.puzzles_eval import solves_line
    fen = "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35"
    moves = "e5f6 e8e1 g1f2 e1f1".split()
    # an oracle that knows the solution for each position it is shown
    board, answer = chess.Board(fen), {}
    for ply, uci in enumerate(moves):
        if ply % 2 == 1:
            answer[board.fen()] = uci
        board.push(chess.Move.from_uci(uci))
    asked = []

    def oracle(b):
        asked.append(b.fen())
        return answer[b.fen()]

    assert solves_line(chess.Board(fen), moves, oracle)
    assert len(asked) == 2                   # asked exactly the two solver moves
    assert not solves_line(chess.Board(fen), moves,
                           lambda b: next(m.uci() for m in b.legal_moves
                                          if m.uci() != answer[b.fen()]))


def test_any_mate_in_one_counts_as_solved():
    from src.puzzles_eval import solves_line
    fen, moves = "6k1/5ppp/8/8/8/8/8/RR4K1 b - - 0 1", ["g8h8", "a1a8"]
    board = chess.Board(fen)
    board.push(chess.Move.from_uci("g8h8"))
    for mate in ("a1a8", "b1b8"):                       # the fixture has two mates
        b = board.copy(); b.push(chess.Move.from_uci(mate)); assert b.is_checkmate()
    assert solves_line(chess.Board(fen), moves, lambda b: "b1b8")
