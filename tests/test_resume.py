"""A training run split across sessions must equal one run straight through.

The 100M-example run is longer than a Kaggle session, so it stops, saves last.pt
and resumes. Weights-only resuming would reset AdamW and restart the LR schedule,
and forgetting the RNG would draw different batches; both are silent. So the test
is bit-exact: split and unsplit runs must end with identical weights AND optimizer
state.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _fake_packed_dataset(d: Path, n: int = 3000) -> None:
    rng = np.random.default_rng(0)
    for split, size in (("train", n), ("val", n // 5)):
        rng.integers(0, 31, (size, 77), dtype=np.uint8).tofile(d / f"{split}_states.bin")
        rng.integers(0, 1968, size, dtype=np.uint16).tofile(d / f"{split}_actions.bin")
        rng.integers(0, 64, size, dtype=np.uint8).tofile(d / f"{split}_buckets.bin")


def _train(data: Path, out: Path, *extra: str) -> None:
    cmd = [sys.executable, "-m", "src.train_chess", "--config", "configs/chess.yaml",
           "--data_dir", str(data), "--device", "cpu", "--amp", "false", "--compile", "false",
           "--n_layer", "1", "--n_head", "2", "--n_embd", "32", "--batch_size", "16",
           "--max_steps", "24", "--warmup_steps", "3", "--eval_interval", "8",
           "--eval_iters", "2", "--log_interval", "8", "--save_every", "0",
           "--out_dir", str(out), *extra]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def _fake_puzzles(d: Path, n: int = 400, k: int = 16) -> None:
    """Packed puzzle arrays in the layout src/puzzle_data.py writes. The values are
    random: the model does not care whether a position is a real one."""
    d.mkdir()
    rng = np.random.default_rng(1)
    for split, size in (("train", n), ("val", n // 4)):
        rng.integers(0, 31, (size, 77), dtype=np.uint8).tofile(d / f"{split}_states.bin")
        rng.integers(0, 1968, (size, k), dtype=np.uint16).tofile(d / f"{split}_cands.bin")
        rng.integers(2, k + 1, size, dtype=np.uint8).tofile(d / f"{split}_ncand.bin")


@pytest.mark.parametrize("with_puzzles", [False, True])
def test_split_run_matches_unsplit_run(tmp_path, with_puzzles):
    """With puzzles too: the ranking batch draws from the same RNG, so a fine-tune
    split across two sessions has to replay it exactly as well."""
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    extra = ()
    if with_puzzles:
        _fake_puzzles(tmp_path / "puzzles")
        extra = ("--puzzle_dir", str(tmp_path / "puzzles"), "--rank_batch", "4")

    _train(data, tmp_path / "straight", *extra)
    _train(data, tmp_path / "part1", "--stop_at_step", "12", *extra)  # stops before an eval step's update
    _train(data, tmp_path / "part2", "--resume", str(tmp_path / "part1" / "last.pt"), *extra)

    straight = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    part1 = torch.load(tmp_path / "part1" / "last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "part2" / "last.pt", weights_only=False)

    assert part1["next_step"] == 12
    assert straight["next_step"] == resumed["next_step"] == 25
    for name, tensor in straight["model"].items():
        assert torch.equal(tensor, resumed["model"][name]), f"weights differ at {name}"
    for key, st in straight["optimizer"]["state"].items():
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(st[field], resumed["optimizer"]["state"][key][field]), field


def test_resume_refuses_a_different_schedule(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _train(data, tmp_path / "part1", "--stop_at_step", "6")

    cmd = [sys.executable, "-m", "src.train_chess", "--config", "configs/chess.yaml",
           "--data_dir", str(data), "--device", "cpu", "--amp", "false",
           "--n_layer", "1", "--n_head", "2", "--n_embd", "32", "--batch_size", "16",
           "--max_steps", "999", "--save_every", "0", "--out_dir", str(tmp_path / "bad"),
           "--resume", str(tmp_path / "part1" / "last.pt")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "LR schedule" in (r.stderr + r.stdout)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_resume_on_an_accelerator_does_not_crash(tmp_path):
    """Regression: resuming on a GPU loaded the whole checkpoint onto the device,
    CPU RNG state included, and torch.set_rng_state refused it. The CPU-only test
    above could never see it; this one runs the same split on MPS, which moves
    tensors off the CPU exactly the way CUDA does. (It checks the resume runs, not
    bit-exactness: MPS kernels are not guaranteed deterministic.)"""
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _train(data, tmp_path / "p1", "--device", "mps", "--stop_at_step", "6")
    cmd = [sys.executable, "-m", "src.train_chess", "--config", "configs/chess.yaml",
           "--data_dir", str(data), "--device", "mps", "--amp", "false", "--compile", "false",
           "--n_layer", "1", "--n_head", "2", "--n_embd", "32", "--batch_size", "16",
           "--max_steps", "24", "--warmup_steps", "3", "--eval_interval", "8",
           "--eval_iters", "2", "--log_interval", "8", "--save_every", "0",
           "--out_dir", str(tmp_path / "p2"), "--resume", str(tmp_path / "p1" / "last.pt")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert torch.load(tmp_path / "p2" / "last.pt", weights_only=False)["next_step"] == 25


def test_init_from_takes_the_weights_and_restarts_the_schedule(tmp_path):
    """Warm restart: same weights, fresh optimizer and fresh LR schedule.

    Used to continue a finished run with more data. --resume would keep the old
    schedule, which has already annealed to min_lr, so training would crawl.
    """
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _train(data, tmp_path / "first")
    source = tmp_path / "first" / "last.pt"
    before = torch.load(source, weights_only=False)

    _train(data, tmp_path / "warm", "--init_from", str(source), "--max_steps", "8")
    warm = torch.load(tmp_path / "warm" / "last.pt", weights_only=False)

    assert warm["next_step"] == 9                      # fresh run, not continued at 25
    assert torch.load(source, weights_only=False)["next_step"] == before["next_step"]
    # the source file is untouched: experiments never overwrite the model they start from
    for name, tensor in before["model"].items():
        assert torch.equal(tensor, torch.load(source, weights_only=False)["model"][name])


def test_init_from_and_resume_are_mutually_exclusive(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _train(data, tmp_path / "first", "--stop_at_step", "6")
    ckpt = str(tmp_path / "first" / "last.pt")
    r = subprocess.run([sys.executable, "-m", "src.train_chess", "--config", "configs/chess.yaml",
                        "--data_dir", str(data), "--device", "cpu", "--amp", "false",
                        "--n_layer", "1", "--n_head", "2", "--n_embd", "32", "--batch_size", "16",
                        "--max_steps", "24", "--out_dir", str(tmp_path / "x"),
                        "--init_from", ckpt, "--resume", ckpt],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "mutually exclusive" in (r.stdout + r.stderr)


def test_init_from_loads_the_source_weights(tmp_path):
    """The test above checks the schedule restarts and the source is untouched -- and
    stayed green with the line that loads the weights deleted. Stopping at step 0,
    before any update, the saved weights must be the source's, bit for bit."""
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _train(data, tmp_path / "first")
    source = torch.load(tmp_path / "first" / "last.pt", weights_only=False)["model"]

    _train(data, tmp_path / "warm", "--init_from", str(tmp_path / "first" / "last.pt"),
           "--stop_at_step", "0")
    _train(data, tmp_path / "cold", "--stop_at_step", "0")
    warm = torch.load(tmp_path / "warm" / "last.pt", weights_only=False)["model"]
    cold = torch.load(tmp_path / "cold" / "last.pt", weights_only=False)["model"]

    for name, tensor in source.items():
        assert torch.equal(tensor, warm[name]), f"{name} was not taken from the source"
    # control: without --init_from the same step-0 save is a fresh initialisation
    assert any(not torch.equal(source[n], cold[n]) for n in source)


def test_the_puzzle_loss_reaches_the_weights(tmp_path):
    """rank_weight 1 and 2 draw identical batches -- same seed, same calls -- and differ
    only in how much the ranking gradient counts. If that loss never reached backward,
    both runs would end bit-identical, which is exactly what the suite could not tell
    before this test. A third run repeats the first to show the runs are deterministic,
    so a difference can only come from the loss."""
    data = tmp_path / "data"
    data.mkdir()
    _fake_packed_dataset(data)
    _fake_puzzles(tmp_path / "puzzles")
    common = ("--puzzle_dir", str(tmp_path / "puzzles"), "--rank_batch", "4",
              "--max_steps", "6")
    for name, weight in (("w1", "1.0"), ("w1_again", "1.0"), ("w2", "2.0")):
        _train(data, tmp_path / name, *common, "--rank_weight", weight)
    w1, again, w2 = (torch.load(tmp_path / n / "last.pt", weights_only=False)["model"]
                     for n in ("w1", "w1_again", "w2"))
    assert all(torch.equal(w1[n], again[n]) for n in w1)
    assert any(not torch.equal(w1[n], w2[n]) for n in w1)
