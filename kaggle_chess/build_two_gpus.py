"""One GPU session that keeps both T4s busy until a fixed quota is spent.

    python kaggle_chess/build_two_gpus.py --quota_h 6.7
    kaggle kernels push -p kaggle_chess_two_gpus_500m/

Two jobs that each need one card, run side by side instead of in two sessions: the
weekly quota counts session time, not cards, so two jobs in one session cost the
quota of one.

- Card 0: the 500M model's light puzzle fine-tune against the 225M model at depth 2,
  on the same 200 starting positions and seed as every match before it. The first N
  positions it reaches pair with results/27 (500M against 225M at depth 2), position by
  position.
- Card 1: search on puzzles, the 500M model and its light fine-tune, each at depth 1
  and depth 2, puzzle by puzzle (src/puzzles_eval.py compare_search). The first 1,000
  give the baseline the fourth search point in the README is missing, and re-measure
  the fine-tune's 770 and 749 of results/26; the rest shrink the noise.

The quota is what decides the length, so both jobs carry deadlines from the session
clock (T0) and a hard stop kills whatever is still running a little later: running out
of quota mid-session ends the session without saving its outputs, which is worse than
stopping early. Both jobs rewrite their results as they go, so a kill loses at most a
position or a hundred puzzles. The match stops earlier than the puzzles because a
position is two games of a minute or more each and is never cut in half. The depth-2
match of results/27 used 8.02h of quota for 8.00h of games, so the time outside T0..end
is a minute or two.

Checkpoints are found by the kernel that wrote them, not by file name: both kernel
sources hold a checkpoints/chess_best.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
W03_KERNEL = "nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03"     # its light fine-tune
DATASET = "nicolangileri/gpt-from-scratch-checkpoints"                  # chess_225M.pt
DIR = "kaggle_chess_two_gpus_500m"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-two-gpus-500m"
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving, a minute or two
MATCH_EARLY_H = 0.10          # a position started just before its deadline still plays
HARD_STOP_AFTER_H = 0.05      # after the end: kill what is left, keep what was written


def data_cell() -> str:
    return '''\
import os, urllib.request
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
BASE = "https://storage.googleapis.com/searchless_chess/data/"
urllib.request.urlretrieve(BASE + "test/action_value_data.bag", DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard the positions come from"
urllib.request.urlretrieve(BASE + "puzzles.csv", DATA + "/puzzles.csv")
'''


def locate_cell() -> str:
    return f'''\
import glob
def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
BASE500 = one("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt")
W03 = one("/kaggle/input/**/{W03_KERNEL.split("/")[1]}/**/chess_best.pt")
M225 = one("/kaggle/input/**/chess_225M.pt")
print("[session] 500M", BASE500, "| w0.3", W03, "| 225M", M225, flush=True)
'''


def run_cell(end_h: float) -> str:
    return f'''\
import os, subprocess, sys, threading, time
END = T0 + {end_h} * 3600
hours_left = lambda: (END - time.time()) / 3600
jobs = {{
    "match": ("0", ["-m", "src.match", "--a", W03, "--b", M225, "--depth", "2",
                    "--positions", "200", "--bag", DATA + "/test.bag", "--device", "cuda",
                    "--threads", "2", "--seed", "0",
                    "--hours", str(hours_left() - {MATCH_EARLY_H}), "--out", "match.json"]),
    "puzzles": ("1", ["-m", "src.puzzles_eval", "--compare", "500M=" + BASE500,
                      "--compare", "500M_w03=" + W03, "--depths", "1", "2",
                      "--puzzles", DATA + "/puzzles.csv", "--num", "10000", "--device", "cuda",
                      "--threads", "2", "--hours", str(hours_left()), "--out", "puzzles.json"]),
}}
print(f"[session] {{(time.time() - T0) / 3600:.2f}}h in, jobs end at {end_h}h", flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

procs, readers = {{}}, []
for name, (gpu, args) in jobs.items():
    procs[name] = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env={{**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}})
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
    (out_dir / "two_gpus_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), data_cell(), locate_cell(), run_cell(end_h)]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "two_gpus_kernel.ipynb", "dataset_sources": [DATASET],
                 "kernel_sources": [BASE_KERNEL, W03_KERNEL]})
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota_h", type=float, required=True,
                    help="GPU hours this session may use; check `kaggle quota` and keep a margin")
    a = ap.parse_args()
    out = build(a.quota_h, ROOT / DIR)
    print(f"wrote {out.name}/: jobs end {a.quota_h - END_BEFORE_QUOTA_H:.2f}h after the session "
          f"starts, hard stop {HARD_STOP_AFTER_H}h later")


if __name__ == "__main__":
    main()
