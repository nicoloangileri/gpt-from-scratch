"""The Kaggle notebook of MCTS 64 against MCTS 32: what would fail on the GPU session."""
import ast
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_mcts_sims_match as sm  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    sm.build(9.0, tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "mcts_sims_match_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta


def test_notebook_parses_and_is_a_new_gpu_kernel(nb):
    cells, meta = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == sm.KERNEL_ID and meta["enable_gpu"] is True
    assert set(meta["kernel_sources"]) == {sm.W1B_KERNEL, sm.ONE_B_KERNEL}
    assert '"src/mcts.py"' in cells[0] and '"src/match.py"' in cells[0]
    assert "T0 + 8.92 * 3600" in cells[1]


def test_both_sides_are_the_bot_at_64_and_32_simulations(nb):
    match = nb[0][-1]
    launched = []
    env = {"time": time, "T0": time.time(), "threading": threading,
           "hours_left": lambda: 8.8, "W1B": "w1b.pt", "ONE_B": "1b.pt", "DATA": "/d",
           "launch": lambda args, card, name: launched.append((card, args)) or (None, None),
           "finish": lambda p, r: 0}
    exec(match, env)
    assert [c for c, _ in launched] == [0, 1]
    for card, args in launched:
        val = lambda f: args[args.index(f) + 1]      # noqa: E731
        assert val("--a_sims") == "64" and val("--b_sims") == "32"
        assert val("--a") == val("--b") == "w1b.pt" and val("--a_leaf") == val("--b_leaf") == "1b.pt"
        assert val("--c_puct") == "0.3" and val("--slice") == f"{card}/2"
        assert "--onepass" not in args and "--depth" not in args


def test_the_match_runner_gives_b_its_own_simulations():
    """--b_sims must reach B: without it B would silently play depth 1."""
    src = (ROOT / "src" / "match.py").read_text()
    assert '"--b_sims"' in src and "mcts_args = ((args.a_sims" in src and "args.b_sims, b_c, b_tau" in src


def test_a_new_pair_gets_its_own_kernel_and_simulations(tmp_path):
    """128 against 64 must not push over results/41's kernel, and must pass 128 and 64."""
    sm.build(11.5, tmp_path, 128, 64)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "nicolangileri/gpt-from-scratch-chess-mcts128-vs-64" != sm.KERNEL_ID
    assert sm.names(128, 64)[0] == "kaggle_chess_mcts128_vs_64"
    assert sm.names(64, 32) == (sm.DIR, sm.KERNEL_ID)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "mcts_sims_match_kernel.ipynb").read_text())["cells"]]
    assert "T0 + 11.42 * 3600" in cells[1]
    launched = []
    env = {"time": time, "T0": time.time(), "threading": threading,
           "hours_left": lambda: 11.3, "W1B": "w1b.pt", "ONE_B": "1b.pt", "DATA": "/d",
           "launch": lambda args, card, name: launched.append((card, args, name)) or (None, None),
           "finish": lambda p, r: 0}
    exec(cells[-1], env)
    for card, args, name in launched:
        val = lambda f: args[args.index(f) + 1]      # noqa: E731
        assert val("--a_sims") == "128" and val("--b_sims") == "64" and val("--tau") == "0.05"
        assert name == f"match_mcts128_vs_mcts64_part{card}"
