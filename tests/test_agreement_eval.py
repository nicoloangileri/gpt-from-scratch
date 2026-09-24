"""The move-quality and calibration metrics, checked on cases whose answers are known.

Unit tests pin each formula to a value computed by hand; the end-to-end test runs
the whole evaluator on a real (tiny) test shard with a stub network, so a metric
wired to the wrong inputs fails too -- a tau computed on (truth, truth) would pass
every unit test and still report 1.0.
"""

import math
import struct

import chess
import numpy as np
import pytest
import torch

from src import agreement_eval
from src.agreement_eval import calibration, kendall_tau_b, mid_pit, position_metrics
from src.chess_data import ACTION_OFFSET, MOVE_TO_ACTION


# --- Kendall's tau-b ---------------------------------------------------------

def test_tau_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    assert kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_tau_on_a_hand_counted_case():
    # pairs (1,2) +, (1,3) +, (2,3) -: (2 - 1) / 3
    assert kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_tau_b_counts_ties_the_way_stockfish_ties_need():
    # x ties its first pair: 2 concordant, 0 discordant; 2 untied x-pairs, 3 y-pairs
    assert kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(6))


def test_tau_is_undefined_when_one_side_does_not_order_anything():
    assert math.isnan(kendall_tau_b([0.99, 0.99, 0.99], [0.1, 0.5, 0.9]))


# --- one position ----------------------------------------------------------

def test_a_different_but_equally_winning_move_is_not_a_miss():
    """Saturation: two moves at Stockfish's top value. The paper's argmax picks the
    first; a model choosing the second is right, and only any_best says so."""
    truth = np.array([0.99, 0.99, 0.20])
    m = position_metrics(np.array([0.5, 0.9, 0.1]), truth)
    assert m["strict"] is False
    assert m["any_best"] is True
    assert m["given_up"] == 0.0


def test_a_real_miss_costs_the_difference():
    m = position_metrics(np.array([0.1, 0.2, 0.9]), np.array([0.8, 0.6, 0.3]))
    assert not m["strict"] and not m["any_best"]
    assert m["given_up"] == pytest.approx(0.5)
    assert m["tau"] == -1.0


# --- calibration -------------------------------------------------------------

def test_mid_pit_on_known_distributions():
    uniform = np.full(4, 0.25)
    assert mid_pit(uniform, 0) == pytest.approx(0.125)
    assert mid_pit(uniform, 3) == pytest.approx(0.875)
    point = np.array([0.0, 1.0, 0.0, 0.0])
    assert mid_pit(point, 1) == pytest.approx(0.5)       # dead centre of its own mass


def test_a_biased_model_shows_its_bias_as_ece():
    truth = np.linspace(0.05, 0.85, 1000)
    pit = np.linspace(0.0005, 0.9995, 1000)              # uniform: perfectly spread
    cal = calibration(truth + 0.1, truth, pit)
    assert cal["mae"] == pytest.approx(0.1)
    assert cal["ece"] == pytest.approx(0.1, abs=1e-9)
    assert cal["cover50"] == pytest.approx(0.5, abs=0.002)
    assert cal["cover90"] == pytest.approx(0.9, abs=0.002)


def test_an_overconfident_distribution_undercovers():
    """Truth always far in the tails of the predicted distribution: the central
    intervals catch almost none of it, however right the mean is."""
    pit = np.r_[np.full(500, 0.01), np.full(500, 0.99)]
    cal = calibration(np.full(1000, 0.5), np.full(1000, 0.5), pit)
    assert cal["ece"] == 0.0                               # the mean is perfect...
    assert cal["cover90"] == 0.0                           # ...and the spread is not


# --- end to end ----------------------------------------------------------------

FEN = "7k/5ppp/8/8/8/8/8/6K1 w - - 0 1"                    # white king: five legal moves


def _write_bag(path, records):
    blob, limits = b"", []
    for fen, move, win_prob in records:
        blob += (bytes([len(fen)]) + fen.encode() + bytes([len(move)]) + move.encode()
                 + struct.pack(">d", win_prob))
        limits.append(len(blob))
    path.write_bytes(blob + b"".join(struct.pack("<q", l) for l in limits)
                     + struct.pack("<Q", len(blob)))


def _stub(value_of_action: dict, buckets: int = 8):
    """A 'network' that puts all its mass on the bucket of a chosen value per move."""
    def model(x):
        logits = torch.full((len(x), buckets), -30.0)
        for row, token in enumerate(x[:, -1].tolist()):
            b = min(int(value_of_action[token] * buckets), buckets - 1)
            logits[row, b] = 30.0
        return logits, None
    return model


@pytest.mark.parametrize("agree", [True, False])
def test_the_evaluator_reads_the_right_numbers(tmp_path, monkeypatch, agree):
    moves = sorted(m.uci() for m in chess.Board(FEN).legal_moves)
    assert len(moves) == 5
    truth = dict(zip(moves, [0.05, 0.30, 0.55, 0.80, 0.95]))
    _write_bag(tmp_path / "t.bag", [(FEN, m, p) for m, p in truth.items()])

    guess = {MOVE_TO_ACTION[m] + ACTION_OFFSET: (p if agree else 1 - p) for m, p in truth.items()}
    centres = torch.tensor((np.arange(8) + 0.5) / 8, dtype=torch.float32)
    monkeypatch.setattr(agreement_eval, "load_model",
                        lambda ckpt, device: (_stub(guess), centres, {"step": 0}))

    out = agreement_eval.run("stub", str(tmp_path / "t.bag"), None, "cpu")
    assert (out["positions"], out["moves_scored"]) == (1, 5)
    if agree:
        assert out["kendall_tau"] == 1.0 and out["action_acc"] == 100.0
        assert out["mean_given_up"] == 0.0
    else:
        assert out["kendall_tau"] == -1.0 and out["action_acc"] == 0.0
        assert out["mean_given_up"] == pytest.approx(0.95 - 0.05)


def test_the_vectorised_pit_matches_the_definition_row_by_row():
    from src.agreement_eval import mid_pits
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(64), size=200)
    buckets = rng.integers(0, 64, size=200)
    expected = [mid_pit(p, int(b)) for p, b in zip(probs, buckets)]
    assert np.allclose(mid_pits(probs, buckets), expected)
