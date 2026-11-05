"""What a ply of search is worth in games: the 500M model at depth 3 against depth 2.

    python kaggle_chess/build_match_depth.py --quota_h 9.0
    kaggle kernels push -p kaggle_chess_match_depth/

Same network on both sides, so the only difference is the search. The positions are the
200 of every match before, each played twice with colours swapped. Each T4 takes every
second position, so a cut at the deadline leaves an even sample of the list.

Depth 2 is what the bot plays, and a depth only counts if it fits the clock, so the
match also records each side's seconds per move: median, tail and worst, on a T4 that
shares its four CPU cores with the other job. The depth-3 side searches one network
call per node with the prefix shared; depth 2 uses the one-pass player, the same moves
in two calls.

The quota decides the length, as in build_two_gpus.py. Both jobs rewrite their results
after every position, so a kill loses at most one position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
DIR = "kaggle_chess_match_depth"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-match-d3-vs-d2"
POSITIONS = 200
CARDS = 2
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
MATCH_EARLY_H = 0.25          # a position started before its deadline still plays
HARD_STOP_AFTER_H = 0.05


def data_cell() -> str:
    return f'''\
import glob, os, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard the positions come from"
hits = glob.glob("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt", recursive=True)
assert len(hits) == 1, hits
BASE500 = hits[0]
print("[session] 500M", BASE500, flush=True)
'''


def run_cell(end_h: float) -> str:
    return f'''\
import os, subprocess, sys, threading, time
END = T0 + {end_h} * 3600
hours = (END - time.time()) / 3600 - {MATCH_EARLY_H}
print(f"[session] {{(time.time() - T0) / 3600:.2f}}h in, jobs end at {end_h}h", flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

procs, readers = {{}}, []
for card in range({CARDS}):
    name = f"card{{card}}"
    procs[name] = subprocess.Popen(
        [sys.executable, "-m", "src.match", "--a", BASE500, "--b", BASE500,
         "--a_depth", "3", "--b_depth", "2", "--onepass", "--positions", "{POSITIONS}",
         "--slice", f"{{card}}/{CARDS}", "--bag", DATA + "/test.bag", "--device", "cuda",
         "--threads", "2", "--seed", "0", "--hours", str(hours),
         "--out", f"match_500M_d3_vs_d2_part{{card}}.json"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONUNBUFFERED": "1"}})
    readers.append(threading.Thread(target=stream, args=(name, procs[name]), daemon=True))
    readers[-1].start()
codes = {{}}
for name, p in procs.items():
    try:
        codes[name] = p.wait(timeout=max(1, END + {HARD_STOP_AFTER_H} * 3600 - time.time()))
    except subprocess.TimeoutExpired:
        p.kill()
        codes[name] = "killed at the hard stop"
for r in readers:
    r.join(timeout=10)
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 1 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "match_depth_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(), run_cell(end_h)]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "match_depth_kernel.ipynb", "dataset_sources": [],
                 "kernel_sources": [BASE_KERNEL]})
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota_h", type=float, required=True,
                    help="GPU hours this session may use; check `kaggle quota` and keep a margin")
    a = ap.parse_args()
    out = build(a.quota_h, ROOT / DIR)
    print(f"wrote {out.name}/: jobs end {a.quota_h - END_BEFORE_QUOTA_H:.2f}h after the session starts")


if __name__ == "__main__":
    main()
