"""The Kaggle notebook of the MCTS match: what would fail on the GPU session."""
import ast
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_mcts_match as mm  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    mm.build(6.5, tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "mcts_match_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta


def test_notebook_parses_and_is_a_new_gpu_kernel(nb):
    cells, meta = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == mm.KERNEL_ID and meta["enable_gpu"] is True
    assert set(meta["kernel_sources"]) == {mm.W1B_KERNEL, mm.ONE_B_KERNEL}
    assert '"src/mcts.py"' in cells[0]


def test_tuning_then_match_with_the_chosen_c(nb, tmp_path, monkeypatch):
    """Run the tuning and match cells with the jobs stubbed: the queues, the choice of c
    and the match command, as the session will run them."""
    _, tune, match = nb[0][1:]
    monkeypatch.chdir(tmp_path)
    launched = []
    solved = {0.03: 1200, 0.1: 1250, 0.3: 1250, 1.0: 1100}

    def launch(args, card, name):
        launched.append((card, args))
        out = args[args.index("--out") + 1]
        if args[0] == "src.mcts":
            c = None if "--depth2" in args else float(args[args.index("--c_puct") + 1])
            json.dump({"solved": 1230 if c is None else solved[c], "attempted": 2000,
                       "seconds_per_move": 0.1}, open(out, "w"))
        return None, None

    import time
    env = {"time": time, "T0": time.time(), "json": json, "threading": threading,
           "hours_left": lambda: 5.0, "W1B": "w1b.pt", "ONE_B": "1b.pt", "DATA": "/d",
           "launch": launch, "finish": lambda p, r: 0}
    exec(tune, env)
    assert env["C"] == 0.1                                   # the tie goes to the smaller c
    assert sorted(c for c, _ in launched) == [0, 0, 0, 1, 1]  # 4 c values + the reference
    launched.clear()
    exec(match, env)
    assert [c for c, _ in launched] == [0, 1]
    for card, args in launched:
        for flag in ("--a_sims", "--c_puct", "--onepass", "--slice"):
            assert flag in args
        assert args[args.index("--c_puct") + 1] == "0.1"
        assert args[args.index("--slice") + 1] == f"{card}/2"
        assert args[args.index("--a") + 1] == args[args.index("--b") + 1] == "w1b.pt"


def test_short_or_long_quota_refused(tmp_path):
    for q in (4.0, 12.0):
        with pytest.raises(SystemExit):
            mm.build(q, tmp_path)
