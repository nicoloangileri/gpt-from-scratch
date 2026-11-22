"""The Kaggle notebook of the 1B search matches: what would fail on the GPU session."""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_1b_search_match as sm  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    sm.build(7.0, tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "search_match_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta


def test_notebook_parses_and_is_a_new_gpu_kernel(nb):
    cells, meta = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == sm.KERNEL_ID and meta["enable_gpu"] is True
    assert set(meta["kernel_sources"]) == {sm.W1B_KERNEL, sm.ONE_B_KERNEL}


def test_a_is_always_the_bot(nb):
    start = nb[0][1]
    for flag in ('"--a", W1B', '"--a_leaf", ONE_B', '"--a_depth", "2"', '"--onepass"',
                 '"--seed", "0"', f'"--positions", "{sm.POSITIONS}"'):
        assert flag in start, flag


def test_the_two_matches_and_their_files(nb):
    cell = nb[0][2]
    assert [m[3] for m in sm.MATCHES] == [2, 1]
    assert '("1B_hybrid_vs_1B_d2", ONE_B, ONE_B, 2)' in cell
    assert '("1B_hybrid_vs_1B_d1", ONE_B, ONE_B, 1)' in cell
    assert 'f"match_{name}_part{c}.json", f"{c}/2"' in cell
    # the list itself, evaluated as the session will: one 4-tuple per match
    line = next(x for x in cell.splitlines() if x.startswith("SPECS = "))
    specs = eval(line.split("=", 1)[1], {"ONE_B": "1b.pt"})
    assert specs == [(n, "1b.pt", "1b.pt", d) for n, _, _, d in sm.MATCHES]


def test_short_or_long_quota_refused(tmp_path):
    for q in (4.0, 12.0):
        with pytest.raises(SystemExit):
            sm.build(q, tmp_path)
