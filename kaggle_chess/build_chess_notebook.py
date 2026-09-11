"""Assemble the Kaggle GPU notebook for the chess action-value model.

Embeds src/ + configs/chess.yaml (base64, byte-identical), then the notebook
pulls one ChessBench shard (1.34 GB, ~17M annotated positions), packs it,
trains, and reports Lichess puzzle accuracy.

    python kaggle_chess/build_chess_notebook.py --steps 4000 --records 6000000
    kaggle kernels push -p kaggle_chess/

The control run and the long run differ only in those two numbers.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "chess_kernel.ipynb"

SOURCES = [
    "src/__init__.py", "src/model.py", "src/data.py", "src/train.py",
    "src/chess_data.py", "src/train_chess.py", "src/puzzles_eval.py",
    "src/bench_chess.py", "src/kernels/__init__.py", "src/kernels/layernorm.py",
    "src/kernels/bench_layernorm.py", "tests/test_layernorm_triton.py",
    "configs/chess.yaml",
]


def _cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def build(steps: int, records: int, batch_size: int, puzzles: int,
          bench: bool = False, compile_model: bool = False, ab: bool = False,
          shards: int = 1, triton: bool = False,
          n_layer: int = 8, n_head: int = 8, n_embd: int = 384, lr: float = 1e-3) -> None:
    blob = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in SOURCES}

    setup = f'''\
import base64, os
os.system("pip install -q chess")
_FILES = {json.dumps(blob)}
for path, b64 in _FILES.items():
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("wrote", list(_FILES))
'''

    data = f'''\
import os, time, urllib.request
# Corpus goes to /kaggle/temp so it stays out of the downloadable kernel output.
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
SHARDS = {shards}
BASE = "https://storage.googleapis.com/searchless_chess/data/"

t, bags = time.time(), []
for i in range(SHARDS):
    bag = DATA + f"/action_value-{{i:05d}}.bag"
    if not os.path.exists(bag):
        urllib.request.urlretrieve(
            BASE + f"train/action_value-{{i:05d}}-of-02148_data.bag", bag)
    bags.append(bag)
urllib.request.urlretrieve(BASE + "puzzles.csv", DATA + "/puzzles.csv")
gb = sum(os.path.getsize(b) for b in bags) / 1e9
print(f"[data] {{SHARDS}} shard(s) downloaded in {{time.time()-t:.0f}}s ({{gb:.2f}} GB)")

t = time.time()
from src.chess_data import prepare_chessbench
prepare_chessbench(bags, DATA + "/chess", max_records={records}, val_records=200_000)
print(f"[data] packed in {{time.time()-t:.0f}}s")
for b in bags:
    os.remove(b)
'''

    bench_cell = f'''\
import sys
sys.argv = ["bench", "--data_dir", DATA + "/chess", "--steps", "200",
            "--n_layer", "12", "--n_head", "10", "--n_embd", "640"]
from src.bench_chess import main
main()
'''

    train = f'''\
import sys
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--max_steps", "{steps}", "--batch_size", "{batch_size}",
            "--compile", "{str(compile_model).lower()}",
            "--n_layer", "{n_layer}", "--n_head", "{n_head}", "--n_embd", "{n_embd}",
            "--lr", "{lr}", "--min_lr", "{lr / 10}",
            "--warmup_steps", "{max(100, steps // 40)}",
            "--eval_interval", "{max(200, steps // 20)}", "--eval_iters", "50",
            "--log_interval", "{max(100, steps // 40)}",
            "--data_dir", DATA + "/chess"]
from src.train_chess import main
main()
'''

    evaluate = f'''\
from src.puzzles_eval import run
run("checkpoints/chess_best.pt", DATA + "/puzzles.csv", {puzzles}, "cuda")
'''

    # Level 3: prove the kernel correct, then time it, then see what it does to
    # a real training step. In that order -- a fast wrong kernel is worthless.
    triton_cell = f'''\
import subprocess, sys, time
print(subprocess.run([sys.executable, "-m", "pytest", "tests/test_layernorm_triton.py",
                      "-q", "--no-header"], capture_output=True, text=True).stdout[-2500:])

from src.kernels.bench_layernorm import main as bench_ln
bench_ln()

import importlib
from src.puzzles_eval import run
for triton_ln in ("false", "true"):
    print("=" * 60, f"\\ntriton_ln = {{triton_ln}}", flush=True)
    sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
                "--max_steps", "{steps}", "--batch_size", "{batch_size}",
                "--compile", "false", "--triton_ln", triton_ln,
                "--warmup_steps", "100", "--eval_interval", "{max(500, steps)}",
                "--eval_iters", "25", "--log_interval", "{max(100, steps // 4)}",
                "--out_dir", f"checkpoints/ln_{{triton_ln}}",
                "--data_dir", DATA + "/chess"]
    import src.train_chess as tc
    importlib.reload(tc)
    t = time.time()
    tc.main()
    print(f"[wall] {{triton_ln}}: {{time.time()-t:.0f}}s for {steps} steps", flush=True)
'''

    # A/B on attention: identical budget, one variable changed.
    ab_cell = f'''\
import sys, importlib
from src.puzzles_eval import run
for causal in ("true", "false"):
    tag = "causal" if causal == "true" else "full"
    print("=" * 60, f"\\nATTENTION = {{tag}}", flush=True)
    sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
                "--max_steps", "{steps}", "--batch_size", "{batch_size}",
                "--compile", "{str(compile_model).lower()}", "--causal", causal,
                "--warmup_steps", "{max(100, steps // 40)}",
                "--eval_interval", "{max(200, steps // 10)}", "--eval_iters", "50",
                "--log_interval", "{max(100, steps // 10)}",
                "--out_dir", f"checkpoints/{{tag}}",
                "--data_dir", DATA + "/chess"]
    import src.train_chess as tc
    importlib.reload(tc)
    tc.main()
    run(f"checkpoints/{{tag}}/chess_best.pt", DATA + "/puzzles.csv", {puzzles}, "cuda")
'''

    nb = {
        "cells": ([_cell(setup), _cell(data), _cell(triton_cell)] if triton
                  else [_cell(setup), _cell(data), _cell(ab_cell)] if ab
                  else [_cell(setup), _cell(data), _cell(bench_cell)] if bench
                  else [_cell(setup), _cell(data), _cell(train), _cell(evaluate)]),
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} (steps={steps}, records={records:,}, batch={batch_size})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--records", type=int, default=6_000_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--puzzles", type=int, default=500)
    ap.add_argument("--bench", action="store_true", help="throughput bench only")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ab", action="store_true", help="A/B causal vs full attention")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--triton", action="store_true", help="Level 3: kernel tests + bench")
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    build(a.steps, a.records, a.batch_size, a.puzzles, a.bench, a.compile, a.ab,
          a.shards, a.triton, a.n_layer, a.n_head, a.n_embd, a.lr)
