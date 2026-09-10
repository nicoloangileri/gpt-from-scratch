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


def test_puzzle_lines_start_with_the_solver_move():
    """ChessBench's puzzles.csv is pre-advanced: Moves[0] is already ours.

    Regression guard: reading it as raw Lichess (opponent moves first) scores the
    model on the opponent's forced replies instead of the key moves, which quietly
    divides the reported accuracy by ~5.
    """
    fen = "4r1k1/2p1qpp1/3p4/1p1P2PQ/1P5b/3R3P/2PBr3/5RK1 b - - 6 26"
    moves = "h4f2 f1f2 e2f2 g1f2".split()
    first_san_of_solution = "Bf2+"

    board = chess.Board(fen)
    assert board.san(chess.Move.from_uci(moves[0])) == first_san_of_solution
