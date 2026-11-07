"""The 1B run's plumbing, checked where a mistake costs nothing: a pack overlapping data
already seen, a session mounting the wrong output, the queue touching the 500M run's files."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_1b as b  # noqa: E402
import build_500m as old  # noqa: E402
import queue_1b as q  # noqa: E402


def test_the_packs_are_new_disjoint_contiguous_and_fit():
    shards = [s for pack in b.PACKS.values() for s in pack]
    assert sorted(shards) == list(range(42, 122)) and len(set(shards)) == len(shards)
    assert not set(shards) & set(old.SHARD_RECORDS) and 0 not in shards   # new, not val
    assert set(b.SHARD_RECORDS) == set(shards)
    assert sum(b.SHARD_RECORDS.values()) >= b.EXAMPLES                    # one pass, no repeats
    for name, pack in b.PACKS.items():
        gb = sum(b.SHARD_RECORDS[s] for s in pack) * 80 / 1e9             # 77 + 2 + 1 bytes each
        assert gb < 12.5 * 2, f"pack {name}: {gb:.1f} GB does not fit two ranks' RAM"
        assert gb < 19.5, f"pack {name}: {gb:.1f} GB, Kaggle keeps 20 GB of output"


def test_the_schedule_is_the_500M_runs_over_more_examples():
    assert b.MAX_STEPS == 488_281 and b.WARMUP == b.MAX_STEPS // 40
    assert (b.LR, b.MIN_LR, b.BATCH) == (old.LR, old.MIN_LR, old.BATCH)


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(old, "ROOT", tmp_path)
    (tmp_path / "kaggle_chess").mkdir()
    (tmp_path / "kaggle_chess" / "kernel-metadata.json").write_text(
        (ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    monkeypatch.setattr(b, "setup_cell", lambda: "T0 = 0\n")
    return tmp_path


def _code(folder: Path, name: str) -> str:
    nb = json.loads((folder / name).read_text())
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_session_one_starts_from_the_500M_model_and_later_ones_from_the_last(built):
    s1 = b.build_session(1, 11.25, "c")
    m1 = json.loads((s1 / "kernel-metadata.json").read_text())
    assert m1["kernel_sources"] == [b.pack_id("c"), b.INIT_KERNEL] and m1["dataset_sources"] == []
    assert m1["enable_gpu"] and m1["machine_shape"] == "NvidiaTeslaT4"    # = two T4s
    code = _code(s1, "session_kernel.ipynb")
    assert '"--init_from", found[0]' in code and '"--resume"' not in code
    assert '"--max_steps", "488281"' in code and "gpt-from-scratch-chess-500m-s3" in code

    s3 = b.build_session(3, 7.1, "e")
    m3 = json.loads((s3 / "kernel-metadata.json").read_text())
    assert m3["kernel_sources"] == [b.pack_id("e"), b.session_id(2)]
    code = _code(s3, "session_kernel.ipynb")
    assert '"--resume", found[0]' in code and "--init_from" not in code
    assert "left_h = 7.1 -" in code

    with pytest.raises(SystemExit):
        b.build_session(2, 12.0, "c")
    with pytest.raises(SystemExit):
        b.build_session(2, 4.0, "a")                                      # a 500M pack


def test_packs_are_cpu_kernels_and_check_their_counts(built):
    for name, shards in b.PACKS.items():
        d = b.build_pack(name)
        meta = json.loads((d / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is False and "machine_shape" not in meta   # no GPU quota
        code = _code(d, "pack_kernel.ipynb")
        assert f"== {sum(b.SHARD_RECORDS[s] for s in shards)}" in code


def test_no_kernel_id_can_replace_an_output_of_the_500M_run():
    ids = {b.session_id(k) for k in range(1, 9)} | {b.pack_id(n) for n in b.PACKS}
    assert not ids & ({old.session_id(k) for k in range(1, 9)} | {old.pack_id(n) for n in old.PACKS})
    assert not any("500m" in i for i in ids)


def test_the_queue_is_wired_to_the_1b_run_and_not_the_old_one():
    assert q.b is b and q.LOG.name == "kaggle_1b.log" and q.OUT.name == "kaggle_1b_out"
    src = (ROOT / "kaggle_chess" / "queue_1b.py").read_text()
    assert "chess_500M" not in src and "500m" not in src and '"chess_1B.pt"' in src
    state = {"sessions": [], "consumed": {n: 0 for n in b.PACKS}}
    assert q.pick_pack(state) == "c"
    size = {n: sum(b.SHARD_RECORDS[s] for s in p) for n, p in b.PACKS.items()}
    state["consumed"]["c"] = size["c"]
    assert q.pick_pack(state) == "d"
