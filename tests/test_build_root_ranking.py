"""The two new Kaggle notebooks: what would only fail once a session is paid for.

A module the notebook imports but does not embed, a cell that does not parse, the
GPU job reading a file the packing cell rewrites, a kernel id that would overwrite an
earlier run's output: each of these costs quota on Kaggle and nothing here.
"""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_int8_cpu as cpu  # noqa: E402
import build_long_run as lr  # noqa: E402
import build_root_ranking as rr  # noqa: E402

EARLIER_KERNELS = {
    "nicolangileri/gpt-from-scratch-chess", "nicolangileri/gpt-from-scratch-chess-500m-s3",
    "nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03",
    "nicolangileri/gpt-from-scratch-chess-two-gpus-500m",
}


def _cells(nb_path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(nb_path.read_text())["cells"]]


def test_every_relative_import_of_an_embedded_module_is_embedded():
    """search.py imports fast_eval: a notebook without it dies at the first search."""
    embedded = set(lr.SOURCES)
    for path in (p for p in lr.SOURCES if p.endswith(".py")):
        for mod in re.findall(r"^\s*from \.(\w+) import", (ROOT / path).read_text(), re.M):
            base = f"{Path(path).parent}/{mod}"
            assert f"{base}.py" in embedded or f"{base}/__init__.py" in embedded, (path, mod)


@pytest.fixture
def root_nb(tmp_path):
    rr.build(6.0, tmp_path)
    return tmp_path


def test_root_ranking_notebook_parses_and_is_a_new_kernel(root_nb):
    cells = _cells(root_nb / "root_ranking_kernel.ipynb")
    for c in cells:
        ast.parse(c)
    meta = json.loads((root_nb / "kernel-metadata.json").read_text())
    assert meta["id"] == rr.KERNEL_ID and meta["id"] not in EARLIER_KERNELS
    assert meta["enable_gpu"] and set(meta["kernel_sources"]) == {rr.BASE_KERNEL, rr.W03_KERNEL}


def test_card_one_never_reads_the_file_the_packing_rewrites(root_nb):
    cells = _cells(root_nb / "root_ranking_kernel.ipynb")
    start, packing = cells[1], cells[2]
    assert '"CUDA_VISIBLE_DEVICES": "1"' in start
    assert "puzzles_root.csv" in start and "puzzles_root.csv" not in packing
    assert '"/puzzles.csv"' in packing


def test_the_fine_tune_is_w03_without_the_puzzles(root_nb):
    train = _cells(root_nb / "root_ranking_kernel.ipynb")[3]
    assert "--puzzle_dir" not in train and "--rank_weight" not in train
    for flag, value in (("--max_steps", "12000"), ("--batch_size", "1024"), ("--lr", "0.0001"),
                        ("--min_lr", "1e-05"), ("--warmup_steps", "240")):
        assert f'"{flag}", "{value}"' in train, flag
    assert '"--init_from", INIT' in train and "INIT = BASE500" in train


def test_a_quota_too_short_for_the_fine_tune_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        rr.build(3.0, tmp_path)


def test_int8_notebook_is_a_cpu_kernel(tmp_path):
    cpu.build(tmp_path)
    for c in _cells(tmp_path / "int8_cpu_kernel.ipynb"):
        ast.parse(c)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False and "machine_shape" not in meta
    assert meta["id"] == cpu.KERNEL_ID and meta["id"] not in EARLIER_KERNELS | {rr.KERNEL_ID}
