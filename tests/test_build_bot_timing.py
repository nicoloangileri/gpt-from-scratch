"""The Kaggle notebook that times the bot's searches on a CPU: what would fail there."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_bot_timing as bt  # noqa: E402


def test_notebook_is_a_new_cpu_kernel_without_a_token(tmp_path):
    bt.build(tmp_path)
    cells = ["".join(c["source"]) for c in
             json.loads((tmp_path / "bot_timing_kernel.ipynb").read_text())["cells"]]
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    for c in cells:
        ast.parse(c)
        assert "LICHESS_TOKEN" not in c and "get_secret" not in c
    assert meta["id"] == bt.KERNEL_ID and meta["enable_gpu"] is False
    assert set(meta["kernel_sources"]) == {bt.W1B_KERNEL, bt.ONE_B_KERNEL}
    for path in ("src/mcts.py", "src/lichess_bot.py", "src/match.py", "src/search.py"):
        assert f'"{path}"' in cells[0], path
    assert "bot.pick(" in cells[1] and "LichessBot.__new__" in cells[1]
