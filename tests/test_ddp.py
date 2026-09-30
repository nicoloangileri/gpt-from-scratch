"""Data-parallel training (the 500M run on Kaggle's two T4s), checked on the CPU.

Two gloo ranks stand in for two GPUs: the code path is the same up to the backend.
What can go wrong silently is the data -- two ranks reading the same rows, or rows
nobody reads -- and the hand-over between sessions; what can go wrong loudly is a
stop that one rank takes and the other does not, which leaves it waiting forever.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.train_chess import ActionValueStream

ROOT = Path(__file__).resolve().parents[1]


def _row_id(states: np.ndarray) -> np.ndarray:
    """Each fake row carries a unique id in its first four FEN tokens (base 31)."""
    s = states[:, :4].astype(np.int64)
    return ((s[:, 0] * 31 + s[:, 1]) * 31 + s[:, 2]) * 31 + s[:, 3]


def _part(d: Path, split: str, n: int, seed: int) -> list[int]:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 31, (n, 77), dtype=np.uint8)
    ids = np.arange(n) + seed * 10_000
    for col in range(3, -1, -1):
        states[:, col] = ids % 31
        ids = ids // 31
    states.tofile(d / f"{split}_states.bin")
    rng.integers(0, 1968, n, dtype=np.uint16).tofile(d / f"{split}_actions.bin")
    rng.integers(0, 64, n, dtype=np.uint8).tofile(d / f"{split}_buckets.bin")
    return list(_row_id(states))


def _packed(root: Path, sizes=(1500, 1101, 7)) -> list[int]:
    """A pack the way the 500M pack kernels write it: val at the root, one folder per
    shard, and the empty files a packer leaves for the split it had nothing for."""
    ids = []
    for i, n in enumerate(sizes):
        ids += list(_part(root / f"shard_{i:05d}", "train", n, seed=i + 1))
        for name in ("states", "actions", "buckets"):
            (root / f"shard_{i:05d}" / f"val_{name}.bin").write_bytes(b"")
    _part(root, "val", 300, seed=5)
    return ids


@pytest.mark.parametrize("world", [2, 3])
def test_rank_slices_are_disjoint_and_cover_the_data(tmp_path, world):
    ids = _packed(tmp_path)
    seen = []
    for rank in range(world):
        s = ActionValueStream(tmp_path, "train", max_ram_gb=1, rank=rank, world=world)
        assert s.in_ram and s.n == len(s.actions) == len(s.states) == len(s.buckets)
        seen.append(list(_row_id(s.states)))
    flat = [a for part in seen for a in part]
    assert sorted(flat) == sorted(ids)                   # every row, once
    assert len(set(flat)) == len(flat)
    assert max(len(p) for p in seen) - min(len(p) for p in seen) <= len(ids) // 100 + 3


def test_rows_stay_whole_across_a_slice(tmp_path):
    """A slice must cut between samples: a state paired with another row's action and
    bucket would train on garbage without any error."""
    _packed(tmp_path)
    whole = ActionValueStream(tmp_path, "train", max_ram_gb=1, rank=0, world=1)
    full = {int(i): (int(a), int(b))
            for i, a, b in zip(_row_id(whole.states), whole.actions, whole.buckets)}
    s = ActionValueStream(tmp_path, "train", max_ram_gb=1, rank=1, world=2)
    for i, a, b in zip(_row_id(s.states), s.actions, s.buckets):
        assert full[int(i)] == (int(a), int(b))


def test_val_is_found_at_the_root_and_empty_files_are_not_parts(tmp_path):
    _packed(tmp_path)
    val = ActionValueStream(tmp_path, "val", max_ram_gb=1)
    assert val.n == 300


def test_a_rank_that_would_not_fit_in_ram_says_so(tmp_path):
    _packed(tmp_path)
    with pytest.raises(MemoryError):
        ActionValueStream(tmp_path, "train", max_ram_gb=1e-6, rank=0, world=2)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(data: Path, out: Path, *extra: str, nproc: int = 2, check: bool = True):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
           "--nproc_per_node", str(nproc), "--master_addr", "127.0.0.1",
           "--master_port", str(_free_port()), "-m", "src.train_chess",
           "--config", "configs/chess.yaml", "--data_dir", str(data), "--device", "cpu",
           "--amp", "false", "--compile", "false", "--n_layer", "1", "--n_head", "2",
           "--n_embd", "32", "--batch_size", "16", "--max_steps", "24",
           "--warmup_steps", "3", "--eval_interval", "8", "--eval_iters", "2",
           "--log_interval", "8", "--save_every", "0", "--out_dir", str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True,
                          timeout=300, env={**os.environ, "OMP_NUM_THREADS": "1",
                                            "PYTHONPATH": str(ROOT)})


def test_two_ranks_split_across_sessions_match_one_straight_run(tmp_path):
    """The resume guarantee of tests/test_resume.py, on two ranks: the RNG state rank 0
    saves must be every rank's, or session two draws different batches."""
    _packed(tmp_path / "data")
    _torchrun(tmp_path / "data", tmp_path / "straight")
    _torchrun(tmp_path / "data", tmp_path / "part1", "--stop_at_step", "12")
    _torchrun(tmp_path / "data", tmp_path / "part2", "--resume", str(tmp_path / "part1" / "last.pt"))
    straight = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "part2" / "last.pt", weights_only=False)
    assert straight["next_step"] == resumed["next_step"] == 25
    for name, tensor in straight["model"].items():
        assert torch.equal(tensor, resumed["model"][name]), f"weights differ at {name}"
    for key, st in straight["optimizer"]["state"].items():
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(st[field], resumed["optimizer"]["state"][key][field]), field


def test_each_rank_of_a_real_run_loads_only_its_slice(tmp_path):
    """The slicing tested above, inside an actual two-rank run. A rank that ignored its
    slice would load everything: nothing fails, twice the RAM is used, and both ranks
    -- seeded alike -- draw identical batches, so half the data is never trained on."""
    sizes = (1500, 1101, 7)
    _packed(tmp_path / "data", sizes)
    r = _torchrun(tmp_path / "data", tmp_path / "out", "--max_steps", "2")
    rank0 = sum(n // 2 for n in sizes)                    # rows [0, n*1//2) of each part
    assert f"[data] train {rank0:,} |" in r.stdout, r.stdout[:400]


def test_a_deadline_stops_every_rank_together(tmp_path):
    """A deadline already past: both ranks must stop at the first vote and exit, not
    one leave and the other wait in an all-reduce until the timeout kills the test."""
    _packed(tmp_path / "data")
    r = _torchrun(tmp_path / "data", tmp_path / "out", "--deadline_s", "0")
    assert "[stopped] before step 0 (deadline)" in r.stdout
    assert torch.load(tmp_path / "out" / "last.pt", weights_only=False)["next_step"] == 0


def test_the_global_batch_must_split_evenly(tmp_path):
    _packed(tmp_path / "data")
    r = _torchrun(tmp_path / "data", tmp_path / "out", "--batch_size", "15", check=False)
    assert r.returncode != 0 and "global batch" in r.stdout + r.stderr
