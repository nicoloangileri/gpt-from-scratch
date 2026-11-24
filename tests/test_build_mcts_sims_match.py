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
    assert '"--b_sims"' in src and "args.a_sims if k == 0 else args.b_sims" in src
