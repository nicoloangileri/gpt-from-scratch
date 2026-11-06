"""Which root should sit over the 500M model's leaves: the distilled fine-tune or the plain one.

    python kaggle_chess/build_match_distill.py --quota_h 4.5
    kaggle kernels push -p kaggle_chess_match_distill/

results/33 put the two level on puzzles, 78.15% against 78.24% (paired difference -0.09,
standard error 0.20), and only the plain fine-tune's root has played: the hybrid, +26 Elo
against the 500M model. Same leaves on both sides here, both at depth 2 with the one-pass
player, so the only difference is the root. 800 starting positions, each twice with colours
swapped; the first 400 are the 400 of results/33's match. Each T4 takes every second position,
so a cut at the deadline leaves an even sample of the list.

Decided before the games: the bot changes root only if the 95% interval, clustering the two
games of each position, excludes zero in the distilled root's favour. At about 26 s per
position on a T4 (results/33) the list takes about 3 hours and the interval is about +-14 Elo,
so a smaller difference stays unmeasured and the root that has already played stays.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
W03_KERNEL = "nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03"     # its w0.3 fine-tune
DISTILL_KERNEL = "nicolangileri/gpt-from-scratch-chess-distill-hybrid"  # the distilled one
DIR = "kaggle_chess_match_distill"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-match-distill-vs-hybrid"
POSITIONS = 800
CARDS = 2
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
MATCH_EARLY_H = 0.10          # a position started before its deadline still plays
HARD_STOP_AFTER_H = 0.05


def data_cell() -> str:
    return f'''\
import glob, os, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard the positions come from"

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
BASE500 = one("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt")
W03 = one("/kaggle/input/**/{W03_KERNEL.split("/")[1]}/**/chess_best.pt")
DISTILLED = one("/kaggle/input/**/{DISTILL_KERNEL.split("/")[1]}/**/chess_500M_w03_distill.pt")
print("[session] 500M", BASE500, "| w0.3", W03, "| distilled", DISTILLED, flush=True)
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
        [sys.executable, "-m", "src.match", "--a", DISTILLED, "--a_leaf", BASE500,
         "--b", W03, "--b_leaf", BASE500, "--depth", "2", "--onepass",
         "--positions", "{POSITIONS}", "--slice", f"{{card}}/{CARDS}",
         "--bag", DATA + "/test.bag", "--device", "cuda", "--threads", "2", "--seed", "0",
         "--hours", str(hours), "--out", f"match_distilled_vs_hybrid_part{{card}}.json"],
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
    (out_dir / "match_distill_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(), run_cell(end_h)]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "match_distill_kernel.ipynb", "dataset_sources": [],
                 "kernel_sources": [BASE_KERNEL, W03_KERNEL, DISTILL_KERNEL]})
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
