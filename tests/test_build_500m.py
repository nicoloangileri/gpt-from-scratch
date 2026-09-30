"""The 500M run's plumbing: shard counts, packs, the session chain, the queue's choices.

Each of these fails silently on Kaggle -- a shard counted one short, a pack that
overlaps the validation shard, a session mounting the wrong output -- and a failure
found there costs quota. So they are checked here, where it costs nothing.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from src.chess_data import bag_records, iter_action_value, pack_train_shard, pack_val

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_500m as b  # noqa: E402
import queue_500m as q  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _real_format_bag(path: Path, n: int) -> None:
    """A shard as ChessBench writes it: the records, then one int64 end offset per
    record -- the last of which is also the file's final 8 bytes, the index pointer."""
    moves = ["e2e4", "d2d4", "g1f3", "c2c4"]
    blob, ends = b"", []
    for i in range(n):
        move = moves[i % len(moves)]
        blob += (bytes([len(START_FEN)]) + START_FEN.encode() + bytes([len(move)])
                 + move.encode() + struct.pack(">d", (i % 100) / 100))
        ends.append(len(blob))
    path.write_bytes(blob + b"".join(struct.pack("<q", e) for e in ends))


def test_the_record_count_from_the_index_is_the_number_of_records(tmp_path):
    """Shard 8 has 201,809 records; reading its last 8 bytes as a separate footer gave
    201,808. Summed over shards 0-9 the right count is 82,395,730 -- exactly what the
    225M run packed (82,195,730 train + 200,000 val)."""
    bag = tmp_path / "s.bag"
    _real_format_bag(bag, 37)
    assert bag_records(bag) == len(list(iter_action_value(bag))) == 37


def test_val_is_the_first_records_and_nothing_else(tmp_path):
    bag = tmp_path / "s.bag"
    _real_format_bag(bag, 250)
    pack_val(bag, tmp_path / "val", records=120)
    files = sorted(p.name for p in (tmp_path / "val").iterdir())
    assert files == ["val_actions.bin", "val_buckets.bin", "val_states.bin"]  # no empty train_*
    buckets = np.fromfile(tmp_path / "val" / "val_buckets.bin", dtype=np.uint8)
    assert len(buckets) == 120


def test_a_train_shard_is_checked_against_its_index(tmp_path):
    bag = tmp_path / "s.bag"
    _real_format_bag(bag, 90)
    meta = pack_train_shard(bag, tmp_path / "shard")
    assert meta["train"] == 90 and meta["val"] == 0
    assert not list((tmp_path / "shard").glob("val_*"))                  # no empty val files


def test_the_packs_are_new_disjoint_contiguous_and_fit_an_output():
    shards = [s for pack in b.PACKS.values() for s in pack]
    assert sorted(shards) == list(range(10, 42)) and len(set(shards)) == len(shards)
    assert 0 not in shards                                   # the validation shard
    assert set(b.SHARD_RECORDS) == set(shards)
    total = sum(b.SHARD_RECORDS.values())
    assert total >= b.EXAMPLES                               # one pass needs no repeats
    for name, pack in b.PACKS.items():
        gb = sum(b.SHARD_RECORDS[s] for s in pack) * 80 / 1e9     # 77 + 2 + 1 bytes each
        assert gb < 19.5, f"pack {name}: {gb:.1f} GB, Kaggle keeps 20 GB of output"


def test_the_schedule_covers_the_examples():
    assert b.MAX_STEPS == b.EXAMPLES // b.BATCH == 268_554
    assert b.WARMUP == b.MAX_STEPS // 40 and b.MIN_LR == pytest.approx(b.LR / 10)


def test_session_one_starts_from_the_225M_checkpoint_and_later_ones_from_the_last(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    (tmp_path / "kaggle_chess").mkdir()
    (tmp_path / "kaggle_chess" / "kernel-metadata.json").write_text(
        (ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    monkeypatch.setattr(b, "setup_cell", lambda: "T0 = 0\n")

    s1 = json.loads((b.build_session(1, 11.25) / "kernel-metadata.json").read_text())
    assert s1["dataset_sources"] == [b.CHECKPOINTS_DATASET]
    assert s1["kernel_sources"] == [b.pack_id("a")]
    assert s1["enable_gpu"] and s1["machine_shape"] == "NvidiaTeslaT4"   # = two T4s

    s3 = json.loads((b.build_session(3, 4.0, "a") / "kernel-metadata.json").read_text())
    assert s3["dataset_sources"] == []
    assert s3["kernel_sources"] == [b.pack_id("a"), b.session_id(2)]
    nb = json.loads((tmp_path / "kaggle_chess_500m_s3" / "session_kernel.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"])
    assert '"--resume", found[0]' in code and "--init_from" not in code
    assert "left_h = 4.0 -" in code                  # eval sized to this session, not 11.25

    for pack in b.PACKS:
        meta = json.loads((b.build_pack(pack) / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is False and "machine_shape" not in meta   # no GPU quota
    with pytest.raises(SystemExit):
        b.build_session(2, 12.0)                     # longer than a session may run
    with pytest.raises(SystemExit):
        b.build_session(2, 4.0, "c")


def test_the_queue_draws_from_the_pack_used_least():
    size = {n: sum(b.SHARD_RECORDS[s] for s in p) for n, p in b.PACKS.items()}
    state = {"sessions": [], "consumed": {n: 0 for n in b.PACKS}}
    assert q.pick_pack(state) == "a"
    state["consumed"]["a"] = int(0.9 * size["a"])
    assert q.pick_pack(state) == "b"
    state["consumed"]["b"] = int(0.95 * size["b"])
    assert q.pick_pack(state) == "a"


def test_the_queue_reads_the_quota_table(monkeypatch):
    table = ("resource  used    remaining  total   refreshAt\n"
             "--------  ------  ---------  ------  -------------------\n"
             "GPU       14.15h  15.85h     30.00h  2026-10-03T00:00:00\n"
             "TPU       0.00h   20.00h     20.00h  2026-10-03T00:00:00")
    monkeypatch.setattr(q, "kaggle", lambda *a: table)
    left, reset = q.quota()
    assert left == 15.85 and reset.isoformat() == "2026-10-03T00:00:00+00:00"


@pytest.mark.parametrize("answer, state", [
    ('nicolangileri/x has status "KernelWorkerStatus.RUNNING"', "running"),
    ('nicolangileri/x has status "KernelWorkerStatus.QUEUED"', "running"),
    ('nicolangileri/x has status "KernelWorkerStatus.COMPLETE"', "complete"),
    ('nicolangileri/x has status "KernelWorkerStatus.ERROR"', "failed"),
    ("Cannot access kernel 'nicolangileri/x' (Permission 'kernels.get' was denied).", "absent"),
])
def test_the_queue_never_mistakes_an_existing_session_for_a_new_one(monkeypatch, answer, state):
    """Pushing a kernel id again replaces that session's output -- the checkpoint the
    next session resumes from. So only a session Kaggle has never heard of is pushed."""
    monkeypatch.setattr(q, "status", lambda kernel: answer)
    assert q.session_state(3) == state


def test_the_watchdog_kills_a_silent_run():
    """A rank stuck without an error would hold the session to Kaggle's 12-hour kill."""
    import time
    code = b.run_cell().replace(
        '[sys.executable, "-m", "torch.distributed.run", "--standalone",\n'
        '           "--nproc_per_node", str(NPROC), "-m", "src.train_chess", *args]',
        '[sys.executable, "-c", args[0]]')
    ns = {"NPROC": 2}
    exec(code, ns)
    t = time.time()
    with pytest.raises(AssertionError):
        ns["torchrun"](["import time; print('step 0', flush=True); time.sleep(120)"],
                       silence_min=0.03)
    assert time.time() - t < 30
    ns["torchrun"](["print('done')"], silence_min=0.03)          # a normal run passes
