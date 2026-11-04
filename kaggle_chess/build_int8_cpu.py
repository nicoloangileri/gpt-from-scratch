"""A CPU session (no GPU quota) for the inference levers of src/fast_eval.py.

    python kaggle_chess/build_int8_cpu.py
    kaggle kernels push -p kaggle_chess_int8_cpu/

On the Mac (ARM, qnnpack) int8 made the network three times slower, so whether it
pays has to be measured where the int8 kernels are the fast ones: an x86 server with
fbgemm, which is what Kaggle's CPU sessions are. Two things are measured on the 500M
model:

1. Latency: the four rows of `python -m src.fast_eval` (fp32 and int8, plain forward
   and shared prefix), in milliseconds per position.
2. What int8 costs: the 10,000 puzzles at depth 1 with the prefix shared, fp32 and
   then int8, one 0/1 per puzzle each, so the two are paired. The fp32 pass is also
   the check that the shared prefix on CPU gives the GPU's 64.84%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"
DIR = "kaggle_chess_int8_cpu"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-int8-cpu"


def run_cell() -> str:
    return f'''\
import glob, os, subprocess, sys, time, urllib.request
import torch
print("[cpu]", os.cpu_count(), "cores | quantized engines", torch.backends.quantized.supported_engines,
      flush=True)
hits = glob.glob("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt", recursive=True)
assert len(hits) == 1, hits
CKPT = hits[0]
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/puzzles.csv",
                           "puzzles.csv")
threads = str(os.cpu_count())
subprocess.run([sys.executable, "-m", "src.fast_eval", "--ckpt", CKPT, "--positions", "50",
                "--threads", threads], check=True)
for name, extra in (("fp32", []), ("int8", ["--int8"])):
    t = time.time()
    subprocess.run([sys.executable, "-m", "src.puzzles_eval", "--ckpt", CKPT,
                    "--puzzles", "puzzles.csv", "--num", "10000", "--device", "cpu",
                    "--threads", threads, "--shared", *extra,
                    "--out", f"puzzles_500M_{{name}}_cpu.json"], check=True)
    print(f"[cpu] {{name}} puzzles in {{time.time() - t:.0f}}s", flush=True)
'''


def build(out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "int8_cpu_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), run_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "int8_cpu_kernel.ipynb", "enable_gpu": False,
                 "dataset_sources": [], "kernel_sources": [BASE_KERNEL]})
    meta.pop("machine_shape", None)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


if __name__ == "__main__":
    print("wrote", build(ROOT / DIR).name + "/")
