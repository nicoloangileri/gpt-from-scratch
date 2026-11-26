"""The Kaggle notebook tuning c and temperature at 64 simulations, then the match."""
import ast
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_mcts_tune64 as t64  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    t64.build(10.4, tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "mcts_tune64_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta, tmp_path


def test_notebook_parses_and_is_a_new_gpu_kernel(nb):
    cells, meta, _ = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == t64.KERNEL_ID and meta["enable_gpu"] is True
    assert set(meta["kernel_sources"]) == {t64.W1B_KERNEL, t64.ONE_B_KERNEL}
    assert '"src/mcts.py"' in cells[0] and '"src/match.py"' in cells[0]
    assert "T0 + 10.32 * 3600" in cells[1]


def test_the_bots_setting_is_in_the_grid_and_the_grid_splits_over_both_cards():
    assert list(t64.BOT) in [list(s) for s in t64.GRID]
    assert len(t64.GRID) == len(set(t64.GRID)) == 6


def _run(cells, tmp_path, monkeypatch, solved):
    """Run the tuning and match cells with fake jobs: each tuning job writes the
    puzzle count `solved` gives its setting."""
    monkeypatch.chdir(tmp_path)
    launched = []

    def launch(args, card, name):
        launched.append((card, args))
        if args[0] == "src.mcts":
            val = lambda f: args[args.index(f) + 1]      # noqa: E731
            c, tau = float(val("--c_puct")), float(val("--tau"))
            Path(val("--out")).write_text(json.dumps(
                {"solved": solved[(c, tau)], "attempted": 2000, "seconds_per_move": 0.4}))
        return None, None

    env = {"time": time, "T0": time.time(), "threading": threading, "json": json,
           "hours_left": lambda: 8.5, "W1B": "w1b.pt", "ONE_B": "1b.pt", "DATA": "/d",
           "launch": launch, "finish": lambda p, r: 0}
    exec(cells[2], env)
    exec(cells[3], env)
    return env, launched


def test_tuning_runs_every_setting_once_at_64_simulations(nb, monkeypatch):
    cells, _, tmp = nb
    solved = {tuple(s): 1700 for s in t64.GRID}
    _, launched = _run(cells, tmp, monkeypatch, solved)
    tunes = [a for _, a in launched if a[0] == "src.mcts"]
    assert len(tunes) == 6 and all(a[a.index("--sims") + 1] == "64" for a in tunes)
    assert sorted(c for c, a in launched if a[0] == "src.mcts") == [0, 0, 0, 1, 1, 1]


def test_the_challenger_is_the_best_other_setting_even_if_the_bots_solves_more(
        nb, monkeypatch):
    cells, _, tmp = nb
    solved = {tuple(s): 1700 for s in t64.GRID}
    solved[(0.3, 0.05)] = 1800                         # the bot's own setting wins puzzles
    solved[(1.0, 0.1)] = 1750
    env, launched = _run(cells, tmp, monkeypatch, solved)
    assert env["CHALLENGER"] == (1.0, 0.1)
    summary = json.loads((tmp / "tune64_summary.json").read_text())
    assert summary["challenger"] == [1.0, 0.1] and summary["bot"] == [0.3, 0.05]


def test_a_tie_goes_to_the_setting_nearest_the_bots(nb, monkeypatch):
    cells, _, tmp = nb
    solved = {tuple(s): 1700 for s in t64.GRID}
    env, _ = _run(cells, tmp, monkeypatch, solved)
    assert env["CHALLENGER"] == (0.3, 0.1)              # smaller c first, as documented


def test_the_match_is_the_challenger_against_the_bots_setting(nb, monkeypatch):
    cells, _, tmp = nb
    solved = {tuple(s): 1700 for s in t64.GRID}
    solved[(0.6, 0.1)] = 1760
    _, launched = _run(cells, tmp, monkeypatch, solved)
    matches = [(c, a) for c, a in launched if a[0] == "src.match"]
    assert [c for c, _ in matches] == [0, 1]
    for card, args in matches:
        val = lambda f: args[args.index(f) + 1]      # noqa: E731
        assert val("--a_sims") == val("--b_sims") == "64"
        assert (val("--c_puct"), val("--tau")) == ("0.6", "0.1")
        assert (val("--b_c_puct"), val("--b_tau")) == ("0.3", "0.05")
        assert val("--a") == val("--b") == "w1b.pt" and val("--a_leaf") == val("--b_leaf") == "1b.pt"
        assert val("--slice") == f"{card}/2" and "--depth" not in args
