"""Assemble a Kaggle GPU training notebook from the local src/ + config.

Embeds model.py/data.py/train.py/sample.py/config.yaml (base64, byte-identical),
then the notebook: installs tiktoken, writes the files, streams a slice of
TinyStories, trains the full 30M model on the T4, generates a text sample and
leaves best.pt in /kaggle/working (the kernel output we pull back).

Run 1 (TinyShakespeare, 334k tokens, 3k steps) overfit hard: val bottomed at 4.41
on step 500 then climbed to 6.90 while train fell to 0.12 -- 74 epochs over a tiny
corpus. Run 2 fixes the cause: a corpus big enough that we stay under one epoch.

    python kaggle_train/build_train_notebook.py
    kaggle kernels push -p kaggle_train/
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "train_kernel.ipynb"

FILES = {
    "src/__init__.py": "",
    "src/model.py": (ROOT / "src/model.py").read_text(),
    "src/data.py": (ROOT / "src/data.py").read_text(),
    "src/train.py": (ROOT / "src/train.py").read_text(),
    "src/sample.py": (ROOT / "src/sample.py").read_text(),
    "configs/config.yaml": (ROOT / "configs/config.yaml").read_text(),
}
BLOB = {k: base64.b64encode(v.encode()).decode() for k, v in FILES.items()}

CELL_SETUP = f'''\
import base64, os
os.system("pip install -q tiktoken")
_FILES = {json.dumps(BLOB)}
for path, b64 in _FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    open(path, "wb").write(base64.b64decode(b64))
print("wrote", list(_FILES))
'''

CELL_DATA = '''\
import os, time, urllib.request
# Keep the corpus out of /kaggle/working: only checkpoints should land in the
# kernel output we download afterwards.
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA + "/tinystories", exist_ok=True)
RAW, MAX_BYTES = DATA + "/tinystories.txt", 400 << 20
URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
       "resolve/main/TinyStories-train.txt")   # 1.92 GB in full; we stream a slice

t = time.time()
req = urllib.request.Request(URL, headers={"User-Agent": "gpt-from-scratch/1.0"})
got = 0
with urllib.request.urlopen(req) as r, open(RAW, "wb") as f:
    while got < MAX_BYTES:
        buf = r.read(1 << 20)
        if not buf:
            break
        f.write(buf); got += len(buf)
print(f"[data] downloaded {got/1e6:.0f} MB in {time.time()-t:.0f}s")

t = time.time()
from src.data import prepare
prepare(RAW, DATA + "/tinystories", val_tokens=1_000_000)
print(f"[data] tokenized in {time.time()-t:.0f}s")
os.remove(RAW)
'''

# Full 30M model on the T4, same size/batch as run 1 so the two are comparable.
# 12000 steps x 32 x 256 = 98M tokens, just under one epoch of the corpus slice.
# dtype stays bf16 in the yaml; train.py detects the T4 (sm_75) and drops to fp16.
CELL_TRAIN = '''\
import sys
sys.argv = ["train", "--config", "configs/config.yaml",
            "--device", "cuda", "--amp", "true", "--compile", "false",
            "--max_steps", "12000", "--warmup_steps", "500",
            "--eval_interval", "500", "--eval_iters", "50", "--log_interval", "250",
            "--data_dir", DATA + "/tinystories"]
from src.train import main
main()
'''

CELL_SAMPLE = '''\
import torch
from src.model import GPT, GPTConfig
from src.data import get_tokenizer
ck = torch.load("checkpoints/best.pt", map_location="cuda")
m = GPT(GPTConfig(**ck["cfg"]))
m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
m.eval().cuda()
enc = get_tokenizer()
ids = torch.tensor([enc.encode_ordinary("Once upon a time")], dtype=torch.long, device="cuda")
out = m.generate(ids, 400, temperature=0.8, top_k=200)
print("="*60, "\\nGENERATED SAMPLE:\\n", enc.decode(out[0].tolist()))
'''


def _cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def build() -> None:
    nb = {
        "cells": [_cell(CELL_SETUP), _cell(CELL_DATA), _cell(CELL_TRAIN), _cell(CELL_SAMPLE)],
        "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
