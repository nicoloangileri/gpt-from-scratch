"""The Kaggle notebook that runs the Lichess bot: what would fail online, or leak."""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_lichess_bot as lb  # noqa: E402


@pytest.fixture
def nb(tmp_path):
    lb.build(tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "lichess_bot_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return cells, meta


def test_notebook_parses_and_is_a_new_cpu_kernel_with_internet(nb):
    cells, meta = nb
    for c in cells:
        ast.parse(c)
    assert meta["id"] == lb.KERNEL_ID and "lichess" in meta["id"]
    assert meta["enable_gpu"] is False and "machine_shape" not in meta
    assert meta["enable_internet"] is True
    assert set(meta["kernel_sources"]) == {lb.W1B_KERNEL, lb.ONE_B_KERNEL}


def test_the_bot_and_its_imports_are_embedded(nb):
    setup = nb[0][0]
    for path in ("src/lichess_bot.py", "src/search.py", "src/puzzles_eval.py", "src/mcts.py"):
        assert f'"{path}"' in setup, path


def test_it_plays_the_configuration_results_37_39_and_41_chose(nb):
    cells = nb[0]
    models, bot = cells[1], cells[-1]
    assert "chess_1B_w03.pt" in models and "gpt-from-scratch-chess-1b-s5" in models
    for flag in ('"--ckpt", ROOT_CKPT', '"--leaf_ckpt", LEAF_CKPT', '"--sims", "64"',
                 '"--c_puct", "0.3"', '"--device", "cpu"', '"--seek"'):
        assert flag in bot, flag


def test_the_token_goes_only_to_the_bots_environment(nb):
    cells = nb[0]
    secret, bot = cells[2], cells[-1]
    assert 'get_secret("LICHESS_TOKEN")' in secret
    assert "print(_TOKEN" not in secret + bot and "len(_TOKEN" not in secret + bot
    assert "{exc}" not in secret                          # an error text could quote it
    assert '"LICHESS_TOKEN": _TOKEN' in bot and "env=env" in bot
    assert "os.environ[" not in bot                       # the notebook's own env is untouched
    for c in cells:
        assert "lip_" not in c and "Bearer" not in c      # no token, no header in the notebook


def test_it_stops_before_kaggle_kills_it(nb):
    bot = nb[0][-1]
    assert lb.STOP_H < lb.LEAVE_H < 12
    assert "signal.SIGTERM" in bot and "if _TOKEN is not None:" in bot


def test_the_session_is_blitz_only(nb):
    """The paper's 2054 is blitz against bots: a session that mixed in rapid games would
    not measure the same thing."""
    from src.lichess_bot import parse_clocks, perf_of
    bot = nb[0][-1]
    assert f'"--clocks", "{lb.CLOCKS}"' in bot
    assert {perf_of(*c) for c in parse_clocks(lb.CLOCKS)} == {"blitz"}
