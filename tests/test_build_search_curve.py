"""The Kaggle notebook of the search-compute curve: what would fail on the GPU session."""
import ast
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_search_curve as sc  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    sc.build(10.5, tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "search_curve_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta


def run_curve(cell, hours, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    launched = []
    env = {"time": time, "T0": time.time(), "threading": threading, "os": __import__("os"),
           "json": json, "hours_left": hours, "ONE_B": "1b.pt", "DATA": "/d",
           "one": lambda pattern: pattern,
           "launch": lambda args, card, name: launched.append((card, args)) or (None, None),
           "finish": lambda p, r: 0}
    exec(cell, env)
    return launched, env


def test_notebook_parses_and_is_a_new_gpu_kernel(nb):
    cells, meta = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == sc.KERNEL_ID and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == [sc.CHECKPOINTS_DATASET]
    assert sc.BASE_KERNEL in meta["kernel_sources"] and sc.ONE_B_KERNEL in meta["kernel_sources"]
    assert '"src/mcts.py"' in cells[0] and "T0 + 10.42 * 3600" in cells[1]


def test_every_model_at_every_budget_cheapest_first(nb, tmp_path, monkeypatch):
    launched, env = run_curve(nb[0][-1], lambda: 10.0, tmp_path, monkeypatch)
    assert len(launched) == 4 * len(sc.SIMS)
    for card, models in enumerate(sc.CARDS):
        jobs = [a for c, a in launched if c == card]
        val = lambda a, f: a[a.index(f) + 1]                    # noqa: E731
        sims = [int(val(a, "--sims")) for a in jobs]
        assert sims == sorted(sims) and sims[0] == 0 and sims[-1] == 256
        assert {val(a, "--ckpt") for a in jobs} == {env["CKPT"][m] for m in models}
        for a in jobs:
            assert "--leaf_ckpt" not in a                         # same network everywhere
            assert val(a, "--c_puct") == "0.3" and val(a, "--tau") == "0.05"
            assert val(a, "--num") == "2000" and float(val(a, "--hours")) < 10.0
    assert set(env["CKPT"]) == {"1B", "500M", "225M", "100M"}
    assert "gpt-from-scratch-checkpoints" in env["CKPT"]["225M"]


def test_no_job_starts_past_the_deadline(nb, tmp_path, monkeypatch):
    launched, env = run_curve(nb[0][-1], lambda: 0.01, tmp_path, monkeypatch)
    assert launched == [] and all(v == "not started: deadline" for v in env["codes"].values())
    assert json.load(open(tmp_path / "search_curve_summary.json"))["curve"] == {}
