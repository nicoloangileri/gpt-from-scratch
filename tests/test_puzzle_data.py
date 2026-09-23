"""Packing Lichess puzzles as a ranking task, and the loss that reads the packing.

The puzzles quoted here are real rows of the held-out set, kept as literals so the
tests run with no data files; each one asserts the chess fact it relies on rather
than trusting the comment next to it.
"""

import io

import chess
import numpy as np
import pytest
import torch

from src.chess_data import ACTION_OFFSET, MOVE_TO_ACTION, SEQUENCE_LENGTH
from src.puzzle_data import (PAD_ACTION, candidates, held_out_puzzle_ids,
                             pack_puzzles, solver_positions)

# Mate in one, exactly one mating move: the opponent plays e1e2, the solver mates
# with h4f2.
MATE_ONE = ("01F7L", "r3kb1r/ppp2ppp/8/2p1Pb2/3Pn2q/3B1P2/PPP3PP/RNBQK2R w KQkq - 1 9",
            "e1e2 h4f2")
# Mate in one with TWO mating moves: g7d7 is the recorded solution, d4d7 mates too.
TWO_MATES = ("07zsM", "2rkr3/3n2Q1/p3p1P1/4PB2/3R1P2/q1p1P3/b1R4P/3K4 b - - 3 32",
             "a2b3 g7d7")


def test_the_solver_moves_on_the_odd_plies():
    """Ply 0 is the opponent's mistake. Getting this backwards once invalidated
    every puzzle number in the project for ten days."""
    _, fen, moves = MATE_ONE
    positions = list(solver_positions(fen, moves.split()))
    assert len(positions) == 1
    board, correct = positions[0]
    assert correct == "h4f2"
    # the position handed over is the one AFTER the opponent's move
    expected = chess.Board(fen)
    expected.push(chess.Move.from_uci("e1e2"))
    assert board.fen() == expected.fen()
    # and the labelled move really is mate, so the label is worth learning
    board.push(chess.Move.from_uci(correct))
    assert board.is_checkmate()


def test_every_solver_ply_becomes_an_example():
    fen = chess.STARTING_FEN
    moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6".split()
    positions = list(solver_positions(fen, moves))
    assert [uci for _, uci in positions] == ["e7e5", "b8c6", "g8f6"]


def test_the_correct_move_is_first_and_the_rest_are_legal():
    import random
    _, fen, moves = MATE_ONE
    board, correct = next(iter(solver_positions(fen, moves.split())))
    cand = candidates(board, correct, 16, random.Random(0))
    assert cand[0] == MOVE_TO_ACTION[correct]
    assert len(cand) == 16
    assert len(set(cand)) == 16                       # no duplicates, no repeats of the answer
    legal = {MOVE_TO_ACTION[m.uci()] for m in board.legal_moves}
    assert set(cand) <= legal


def test_alternative_mates_are_not_taught_as_mistakes():
    """Lichess accepts any mate in one, so scoring the other one is not an error."""
    import random
    _, fen, moves = TWO_MATES
    board, correct = next(iter(solver_positions(fen, moves.split())))
    other_mate = chess.Move.from_uci("d4d7")
    board.push(other_mate)
    assert board.is_checkmate()                       # the premise of this test
    board.pop()

    cand = candidates(board, correct, 16, random.Random(0))
    assert cand[0] == MOVE_TO_ACTION[correct]
    assert MOVE_TO_ACTION["d4d7"] not in cand


def test_a_position_with_nothing_to_rank_is_dropped():
    import random
    # only move: the king must recapture
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    assert board.legal_moves.count() == 1
    only = next(iter(board.legal_moves)).uci()
    assert candidates(board, only, 16, random.Random(0)) is None


def _write_csv_zst(path, rows):
    import zstandard
    header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
    body = "".join(f"{pid},{fen},{moves},1500,74,95,100,mateIn1,https://x,\n"
                   for pid, fen, moves in rows)
    path.write_bytes(zstandard.ZstdCompressor().compress((header + body).encode()))


def test_packing_round_trip(tmp_path):
    _write_csv_zst(tmp_path / "p.csv.zst", [MATE_ONE, TWO_MATES])
    stats = pack_puzzles(tmp_path / "p.csv.zst", tmp_path / "out",
                         val_positions=1, log_every=0)
    assert stats["train"] == 1 and stats["val"] == 1

    states = np.fromfile(tmp_path / "out" / "train_states.bin", dtype=np.uint8)
    cands = np.fromfile(tmp_path / "out" / "train_cands.bin", dtype=np.uint16)
    ncand = np.fromfile(tmp_path / "out" / "train_ncand.bin", dtype=np.uint8)
    assert states.reshape(-1, SEQUENCE_LENGTH).shape == (1, SEQUENCE_LENGTH)
    assert cands.reshape(1, -1).shape == (1, 16)
    assert ncand.tolist() == [16]


def test_held_out_puzzles_are_excluded(tmp_path):
    """Training on the benchmark would make every puzzle number in the repo a lie."""
    _write_csv_zst(tmp_path / "p.csv.zst", [MATE_ONE, TWO_MATES])
    held_out = tmp_path / "held_out.csv"
    held_out.write_text("PuzzleId,Rating\n01F7L,827\n")

    stats = pack_puzzles(tmp_path / "p.csv.zst", tmp_path / "out",
                         exclude_ids=held_out_puzzle_ids(held_out),
                         val_positions=0, log_every=0)
    assert stats["excluded"] == 1
    assert stats["train"] == 1                        # only the puzzle that is not held out


def test_short_candidate_rows_are_padded_and_counted(tmp_path):
    """A position with few legal moves still packs; ncand says how far to read."""
    # after the opponent's move the solver has 7 legal moves, so 6 alternatives
    fen = "7k/5ppp/8/8/8/8/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    board.push(chess.Move.from_uci("g1g2"))
    assert board.legal_moves.count() == 7
    rows = [("tiny", fen, "g1g2 h8g8")]               # ply 0 white, ply 1 black: solver
    _write_csv_zst(tmp_path / "p.csv.zst", rows)
    stats = pack_puzzles(tmp_path / "p.csv.zst", tmp_path / "out",
                         val_positions=0, log_every=0)
    assert stats["train"] == 1
    cands = np.fromfile(tmp_path / "out" / "train_cands.bin", dtype=np.uint16)
    ncand = int(np.fromfile(tmp_path / "out" / "train_ncand.bin", dtype=np.uint8)[0])
    assert ncand < 16
    assert (cands[ncand:] == PAD_ACTION).all()        # the tail is padding


# --- the loss --------------------------------------------------------------

def test_listwise_loss_rewards_ranking_the_puzzle_move_first():
    from src.train_chess import listwise_loss
    mask = torch.ones(1, 4, dtype=torch.bool)
    right = listwise_loss(torch.tensor([[0.9, 0.3, 0.2, 0.1]]), mask, 0.05)
    wrong = listwise_loss(torch.tensor([[0.1, 0.9, 0.2, 0.3]]), mask, 0.05)
    assert right < wrong
    assert right < 0.01                               # clearly first: almost no loss


def test_padding_cannot_win():
    """A padded column holds a junk action; if it could be picked, the loss would
    be computed against a move that is not even legal."""
    from src.train_chess import listwise_loss
    values = torch.tensor([[0.5, 0.4, 9.9, 9.9]])     # padding with an absurd value
    mask = torch.tensor([[True, True, False, False]])
    loss = listwise_loss(values, mask, 0.05)
    unpadded = listwise_loss(values[:, :2], mask[:, :2], 0.05)
    assert torch.allclose(loss, unpadded)


def test_temperature_sets_how_much_a_win_probability_gap_counts():
    from src.train_chess import listwise_loss
    values = torch.tensor([[0.55, 0.50]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    # the same 5-point gap is a preference at 0.05 and noise at 1.0
    assert listwise_loss(values, mask, 0.05) < listwise_loss(values, mask, 1.0)


def test_stream_pairs_every_candidate_with_its_position(tmp_path):
    from src.train_chess import PuzzleRankStream
    _write_csv_zst(tmp_path / "p.csv.zst", [MATE_ONE, TWO_MATES])
    pack_puzzles(tmp_path / "p.csv.zst", tmp_path / "out", val_positions=0, log_every=0)

    stream = PuzzleRankStream(tmp_path / "out", "train")
    x, mask = stream.batch(2, "cpu")
    assert x.shape == (2 * stream.k, SEQUENCE_LENGTH + 1)
    assert mask.shape == (2, stream.k)
    # the position is repeated for each candidate, and only the action token changes
    first = x[:stream.k]
    assert (first[:, :SEQUENCE_LENGTH] == first[0, :SEQUENCE_LENGTH]).all()
    assert (first[:, -1] >= ACTION_OFFSET).all()
    assert len(set(first[:, -1].tolist())) == stream.k


def test_rank_step_scores_the_moves_of_one_position_together(tmp_path):
    """End to end on a real model: shapes line up and the accuracy is a fraction."""
    from src.model import GPT, GPTConfig
    from src.train_chess import rank_step, _nullctx
    from src.chess_data import FULL_VOCAB, SAMPLE_LENGTH
    from src.train_chess import PuzzleRankStream

    _write_csv_zst(tmp_path / "p.csv.zst", [MATE_ONE, TWO_MATES])
    pack_puzzles(tmp_path / "p.csv.zst", tmp_path / "out", val_positions=0, log_every=0)
    stream = PuzzleRankStream(tmp_path / "out", "train")

    model = GPT(GPTConfig(vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH,
                          num_classes=8, n_layer=1, n_head=2, n_embd=32))
    centres = torch.tensor((np.arange(8) + 0.5) / 8, dtype=torch.float32)
    cfg = {"rank_batch": 2, "rank_temp": 0.05}
    loss, acc = rank_step(model, centres, stream, cfg, _nullctx(), "cpu")
    assert loss.requires_grad and loss.item() > 0
    assert 0.0 <= acc <= 1.0
    loss.backward()                                   # the gradient reaches the weights
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
