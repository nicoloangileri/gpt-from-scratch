"""Assemble the two Kaggle notebooks for a run longer than one GPU session.

Kaggle stops a session at 12 hours; 100M examples take about 20. So:

  session A  packs the data, trains until a deadline under the limit, saves last.pt
  session B  mounts A's output (kernel_sources), re-packs the SAME shards the same
             way (deterministic, so the arrays are identical), resumes from last.pt,
             finishes and reports puzzle accuracy

Resuming restores weights, AdamW moments, GradScaler, RNG and the schedule position;
tests/test_resume.py checks that a split run ends bit-identical to an unsplit one.

    python kaggle_chess/build_long_run.py --examples 100000000 --shards 6
    kaggle kernels push -p kaggle_chess/              # session A
    # only once A is COMPLETE, and push nothing else to that kernel meanwhile:
    kaggle kernels push -p kaggle_chess_resume/       # session B
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A_DIR = ROOT / "kaggle_chess"
B_DIR = ROOT / "kaggle_chess_resume"
SOURCES = [
    "src/__init__.py", "src/model.py", "src/data.py", "src/train.py",
    "src/chess_data.py", "src/train_chess.py", "src/puzzles_eval.py",
    "src/kernels/__init__.py", "src/kernels/layernorm.py", "configs/chess.yaml",
]
SESSION_HOURS = 11.25          # Kaggle kills at 12h; the rest is margin to save outputs


def _cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def _notebook(cells: list[str]) -> dict:
    return {"cells": [_cell(c) for c in cells],
            "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                        "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def setup_cell() -> str:
    blob = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in SOURCES}
    return f'''\
import base64, os, time
T0 = time.time()                      # the session clock starts here
os.system("pip install -q chess")
_FILES = {json.dumps(blob)}
for path, b64 in _FILES.items():
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("wrote", list(_FILES))
'''


def data_cell(shards: int) -> str:
    return f'''\
import glob, os, time, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
BASE = "https://storage.googleapis.com/searchless_chess/data/"
SHARDS = {shards}

def one_shard_at_a_time():
    """Fetch a shard only when the packer reaches it, and drop the previous one.

    prepare_chessbench reads its shard list lazily, so at most one 1.34 GB bag sits
    on disk next to the growing packed arrays -- ~9.5 GB peak instead of ~16.
    """
    prev = None
    for i in range(SHARDS):
        if prev:
            os.remove(prev)
        bag, t = DATA + f"/action_value-{{i:05d}}.bag", time.time()
        urllib.request.urlretrieve(BASE + f"train/action_value-{{i:05d}}-of-02148_data.bag", bag)
        print(f"[data] shard {{i + 1}}/{{SHARDS}} downloaded in {{time.time() - t:.0f}}s", flush=True)
        prev = bag
        yield bag

urllib.request.urlretrieve(BASE + "puzzles.csv", DATA + "/puzzles.csv")
from src.chess_data import prepare_chessbench
t = time.time()
prepare_chessbench(one_shard_at_a_time(), DATA + "/chess", max_records=None, val_records=200_000)
for leftover in glob.glob(DATA + "/*.bag"):
    os.remove(leftover)
print(f"[data] packed in {{time.time() - t:.0f}}s", flush=True)

# Keep the packed states in RAM only if they fit with room to spare; otherwise the
# loader falls back to memmap on its own, slower but safe.
try:
    avail_gb = next(int(l.split()[1]) for l in open("/proc/meminfo")
                    if l.startswith("MemAvailable")) / 1e6
except (OSError, StopIteration):
    avail_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
states_gb = os.path.getsize(DATA + "/chess/train_states.bin") / 1e9
MAX_RAM_GB = round(0.5 * avail_gb, 1)
print(f"[ram] {{avail_gb:.1f}} GB available, train states {{states_gb:.1f}} GB -> "
      f"{{'in RAM' if states_gb <= MAX_RAM_GB else 'memmap'}}", flush=True)
'''


def train_cell(max_steps: int, batch: int, resume: bool) -> str:
    return f'''\
import sys, time
budget = {SESSION_HOURS} * 3600 - (time.time() - T0)
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--data_dir", DATA + "/chess", "--out_dir", "checkpoints",
            "--max_steps", "{max_steps}", "--batch_size", "{batch}", "--compile", "true",
            "--warmup_steps", "{max(100, max_steps // 40)}", "--max_ram_gb", str(MAX_RAM_GB),
            "--eval_interval", "4000", "--eval_iters", "50", "--log_interval", "1000",
            "--save_every", "2000", "--deadline_s", str(budget)]{' + ["--resume", RESUME]' if resume else ''}
from src.train_chess import main
main()
'''


def locate_cell() -> str:
    return '''\
import glob, os, shutil, torch
found = glob.glob("/kaggle/input/**/last.pt", recursive=True)
assert found, "no last.pt under /kaggle/input: is session A attached as a kernel source?"
RESUME = max(found, key=lambda p: torch.load(p, map_location="cpu", weights_only=False)["next_step"])
os.makedirs("checkpoints", exist_ok=True)
best = os.path.join(os.path.dirname(RESUME), "chess_best.pt")
if os.path.exists(best):                  # keep A's best even if B never beats it
    shutil.copy(best, "checkpoints/chess_best.pt")
print("[resume] using", RESUME, flush=True)
'''


def eval_cell(max_steps: int, puzzles: int) -> str:
    return f'''\
import torch
state = torch.load("checkpoints/last.pt", map_location="cpu", weights_only=False)
if state["next_step"] > {max_steps}:
    from src.puzzles_eval import run
    run("checkpoints/chess_best.pt", DATA + "/puzzles.csv", {puzzles}, "cuda")
else:
    print(f"[session] stopped at step {{state['next_step']}} of {max_steps}: needs one more resume session")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=100_000_000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--puzzles", type=int, default=1000)
    a = ap.parse_args()
    max_steps = a.examples // a.batch_size

    (A_DIR / "chess_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(a.shards), train_cell(max_steps, a.batch_size, resume=False)]), indent=1))

    B_DIR.mkdir(exist_ok=True)
    (B_DIR / "chess_resume_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(a.shards), locate_cell(),
         train_cell(max_steps, a.batch_size, resume=True), eval_cell(max_steps, a.puzzles)]), indent=1))
    meta = json.loads((A_DIR / "kernel-metadata.json").read_text())
    meta.update({"id": "nicolangileri/gpt-from-scratch-chess-resume",
                 "title": "gpt-from-scratch-chess-resume",
                 "code_file": "chess_resume_kernel.ipynb",
                 "kernel_sources": ["nicolangileri/gpt-from-scratch-chess"]})
    (B_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"session A -> {A_DIR/'chess_kernel.ipynb'}")
    print(f"session B -> {B_DIR/'chess_resume_kernel.ipynb'}  (sources nicolangileri/gpt-from-scratch-chess)")
    print(f"{a.examples/1e6:.0f}M examples = {max_steps} steps x {a.batch_size}, {a.shards} shards")


if __name__ == "__main__":
    main()
