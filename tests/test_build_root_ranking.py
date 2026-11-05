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


# --- the distillation + hybrid-match session -------------------------------------
import build_distill_hybrid as dh  # noqa: E402


@pytest.fixture
def dh_nb(tmp_path):
    dh.build(10.5, tmp_path)
    return tmp_path


def test_distill_hybrid_notebook_parses_and_is_a_new_kernel(dh_nb):
    for c in _cells(dh_nb / "distill_hybrid_kernel.ipynb"):
        ast.parse(c)
    meta = json.loads((dh_nb / "kernel-metadata.json").read_text())
    assert meta["id"] == dh.KERNEL_ID
    assert meta["id"] not in EARLIER_KERNELS | {rr.KERNEL_ID, cpu.KERNEL_ID}
    assert meta["enable_gpu"] and set(meta["kernel_sources"]) == {dh.BASE_KERNEL, dh.W03_KERNEL}


def test_the_match_bag_survives_the_packing(dh_nb):
    """The packing cell deletes every .bag in DATA: the match's copy must live elsewhere."""
    cells = _cells(dh_nb / "distill_hybrid_kernel.ipynb")
    start = cells[1]
    packing = [c for c in cells if "prepare_chessbench" in c and "_FILES" not in c][0]
    assert 'glob.glob(DATA + "/*.bag")' in packing
    assert 'MATCH_DIR = DATA + "_match"' in start and '"--bag", MATCH_DIR + "/test.bag"' in start
    assert '"CUDA_VISIBLE_DEVICES": "1"' in start
    for flag in ('"--a", W03', '"--a_leaf", BASE500', '"--b", BASE500', '"--onepass"',
                 '"--depth", "2"', '"--seed", "0"'):
        assert flag in start, flag


def test_the_distilled_fine_tune_is_w03_plus_the_teacher(dh_nb):
    """Against the notebook that produced results/26: the same command, two flags more."""
    def train(nb):
        return [c for c in _cells(nb) if 'sys.argv = ["train_chess"' in c][0]
    new = train(dh_nb / "distill_hybrid_kernel.ipynb")
    old = train(ROOT / "kaggle_chess_puzzle_500m_w03" / "puzzle_kernel.ipynb")
    args = lambda cell: cell[cell.index("sys.argv"):cell.index("from src.train_chess")]
    extra = '            "--distill_from", BASE500, "--distill_weight", "1.0",\n'
    assert args(new).replace(extra, "") == args(old)
    assert "INIT = BASE500" in new


def test_a_quota_too_short_for_the_distilled_fine_tune_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        dh.build(6.0, tmp_path)
