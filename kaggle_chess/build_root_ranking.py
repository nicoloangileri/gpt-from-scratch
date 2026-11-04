"""One GPU session for two questions about the puzzle fine-tune, a card each.

    python kaggle_chess/build_root_ranking.py --quota_h 6.0
    kaggle kernels push -p kaggle_chess_root_ranking/

- Card 1: why depth 2 hurts the fine-tune (src/root_ranking.py). The 500M model and
  its w0.3 fine-tune, every root/leaf pairing at depth 1 and 2, plus where the puzzle
  move ranks by Q and by the backed-up value. Its same-network configurations pair
  with results/30 puzzle by puzzle: that is the check that the one-pass shortcut is
  the search.
- Card 0: the same fine-tune without the ranking loss (w0). Same start (500M s3), same
  three shards, same steps, batch, learning rate and schedule as w0.3, only no puzzle
  batch. Whatever w0.3 gained over the base that w0 also gains belongs to the extra
  steps on fresh positions; what is left belongs to the ranking loss. Then its puzzles
  at depth 1 (one 0/1 per puzzle), its move agreement, and the root/leaf pass for w0
  and the base, which says whether the fine-tune alone already turns search against
  the network.

Card 1 starts first and needs nothing but the puzzles, so it runs while card 0's
shards are packed. It reads its own copy of puzzles.csv: the packing cell downloads
the file again, and rewriting a file another process is reading would hand it a
truncated one.

The quota decides the length, as in build_two_gpus.py: the training deadline leaves
room for the evaluations, every job carries a deadline from the session clock (T0),
and what is still running a little after the end is killed. Both root/leaf passes
rewrite their results every 100 puzzles, so a kill loses at most a hundred.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, data_cell, setup_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
W03_KERNEL = "nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03"     # its w0.3 fine-tune
DIR = "kaggle_chess_root_ranking"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-root-ranking-w0"
SHARDS = 3                    # as the w0.3 fine-tune (results/26)
STEPS, BATCH, LR = 12000, 1024, 1e-4
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
EVAL_RESERVE_H = 1.5          # after training: puzzles ~0.2h, agreement ~0.5h, then the pass
HARD_STOP_AFTER_H = 0.05


def start_cell(end_h: float) -> str:
    """Locate the checkpoints and start card 1 in the background."""
    return f'''\
import glob, os, subprocess, sys, threading, time, urllib.request
END = T0 + {end_h} * 3600
hours_left = lambda: (END - time.time()) / 3600
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
ROOT_PUZZLES = DATA + "/puzzles_root.csv"
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/puzzles.csv",
                           ROOT_PUZZLES)

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
BASE500 = one("/kaggle/input/**/{BASE_KERNEL.split("/")[1]}/**/chess_best.pt")
W03 = one("/kaggle/input/**/{W03_KERNEL.split("/")[1]}/**/chess_best.pt")
print("[session] 500M", BASE500, "| w0.3", W03, flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

ROOT_JOB = subprocess.Popen(
    [sys.executable, "-m", "src.root_ranking", "--model", "500M=" + BASE500,
     "--model", "500M_w03=" + W03, "--puzzles", ROOT_PUZZLES, "--num", "10000",
     "--device", "cuda", "--threads", "2", "--hours", str(hours_left()),
     "--out", "root_ranking_500M_w03.json"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    env={{**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}})
ROOT_READER = threading.Thread(target=stream, args=("root", ROOT_JOB), daemon=True)
ROOT_READER.start()
print(f"[session] card 1 started {{(time.time() - T0) / 3600:.2f}}h in, jobs end at {end_h}h",
      flush=True)
'''


def finetune_cell() -> str:
    """The w0.3 fine-tune's command (build_puzzle_finetune.finetune_cell) without the
    puzzle batch. The notebook sees both cards and "cuda" is the first: card 1's job
    was started with only the second visible."""
    return f'''\
import sys, time
INIT = BASE500
budget = END - {EVAL_RESERVE_H} * 3600 - time.time()
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--data_dir", DATA + "/chess", "--out_dir", "checkpoints",
            "--max_steps", "{STEPS}", "--batch_size", "{BATCH}", "--compile", "true",
            "--lr", "{LR}", "--min_lr", "{LR / 10}",
            "--warmup_steps", "{max(100, STEPS // 50)}", "--max_ram_gb", str(MAX_RAM_GB),
            "--eval_interval", "1000", "--eval_iters", "25", "--log_interval", "250",
            "--save_every", "2000", "--deadline_s", str(budget), "--init_from", INIT]
from src.train_chess import main
main()
'''


def eval_cell() -> str:
    return f'''\
import json, shutil, urllib.request
from src.puzzles_eval import run as puzzles
from src.agreement_eval import run as agreement
from src.root_ranking import run as root_ranking
W0 = "chess_500M_w0.pt"
shutil.copy("checkpoints/chess_best.pt", W0)      # a name that says what it is
results = {{"puzzles": puzzles(W0, DATA + "/puzzles.csv", 10000, "cuda",
                              out="puzzles_500M_w0.json")}}
json.dump(results, open("results_w0.json", "w"), indent=1, default=float)
urllib.request.urlretrieve("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag",
                           DATA + "/test.bag")
results["agreement"] = agreement(W0, DATA + "/test.bag", None, "cuda")
json.dump(results, open("results_w0.json", "w"), indent=1, default=float)
print(f"[session] root/leaf pass for w0 from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
root_ranking({{"500M_w0": W0, "500M": BASE500}}, DATA + "/puzzles.csv", 10000, "cuda",
             hours=hours_left(), out="root_ranking_500M_w0.json")
'''


def wait_cell() -> str:
    return f'''\
import subprocess, time
try:
    code = ROOT_JOB.wait(timeout=max(1, END + {HARD_STOP_AFTER_H} * 3600 - time.time()))
except subprocess.TimeoutExpired:
    ROOT_JOB.kill()
    code = "killed at the hard stop"
ROOT_READER.join(timeout=10)
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: card 1 {{code}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 3.5 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h, "
                         "too short for the fine-tune and its evaluations")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "root_ranking_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), data_cell(SHARDS), finetune_cell(), eval_cell(),
         wait_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "root_ranking_kernel.ipynb", "dataset_sources": [],
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
          f"starts, training at most {a.quota_h - END_BEFORE_QUOTA_H - EVAL_RESERVE_H:.2f}h in")


if __name__ == "__main__":
    main()
