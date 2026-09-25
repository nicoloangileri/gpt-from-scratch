"""Assemble the Kaggle notebook for a long head-to-head match at depth 2.

    python kaggle_chess/build_match_notebook.py
    kaggle kernels push -p kaggle_chess_match_d2/

At depth 1 the 225M model beat its puzzle-fine-tuned self by +103 Elo over 400
games (results/19). At depth 2 a first-to-7 went 7.5-6.5 the other way, over 14
games -- too few to say anything, and 400 games at depth 2 are about 30 hours of
laptop CPU. On a GPU the network calls inside the search are much cheaper.

Same 200 starting positions as the depth-1 match (same shard, same seed), so the
two results compare on identical openings. How long it takes on a T4 has not been
measured, so the match carries a deadline inside the session limit and rewrites
results.json after every position: if it is slower than hoped, it stops cleanly
with what it has, never between the two colours of a position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, SESSION_HOURS, _notebook, setup_cell  # noqa: E402


def data_cell() -> str:
    return '''\
import os, time, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard the positions come from"
'''


def match_cell(a: str, b: str, depth: int, positions: int) -> str:
    return f'''\
import glob, os, sys, time
found = {{os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True)}}
assert "{a}" in found and "{b}" in found, f"attach the checkpoints dataset: {{sorted(found)}}"
hours = {SESSION_HOURS} - 0.5 - (time.time() - T0) / 3600      # leave time to save outputs
print(f"[session] match budget {{hours:.2f}}h", flush=True)
sys.argv = ["match", "--a", found["{a}"], "--b", found["{b}"], "--depth", "{depth}",
            "--positions", "{positions}", "--bag", DATA + "/test.bag", "--device", "cuda",
            "--threads", "4", "--seed", "0", "--hours", str(hours), "--out", "results.json"]
from src.match import main
main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="chess_225M.pt")
    ap.add_argument("--b", default="chess_225M_puzzles.pt")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--dataset", default="nicolangileri/gpt-from-scratch-checkpoints")
    ap.add_argument("--dir", default="kaggle_chess_match_d2")
    ap.add_argument("--id", default="nicolangileri/gpt-from-scratch-chess-match-d2")
    a = ap.parse_args()

    out = ROOT / a.dir
    out.mkdir(exist_ok=True)
    (out / "match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(), match_cell(a.a, a.b, a.depth, a.positions)]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": a.id, "title": a.id.split("/")[-1], "code_file": "match_kernel.ipynb",
                 "dataset_sources": [a.dataset], "kernel_sources": []})
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.dir}/: {a.a} vs {a.b}, depth {a.depth}, {a.positions} positions x 2")


if __name__ == "__main__":
    main()
