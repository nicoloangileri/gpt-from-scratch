"""One GPU session, a card each: the hybrid search in games, and a fine-tune that keeps
its leaves.

    python kaggle_chess/build_distill_hybrid.py --quota_h 10.5
    kaggle kernels push -p kaggle_chess_distill_hybrid/

results/32 found that depth 2 hurts the light puzzle fine-tune because the fine-tune
made the values it backs up worse, and that its root over the 500M model's leaves
solves 78.2% of the puzzles, the best score in the project. Two follow-ups:

- Card 1: does the hybrid also play better? The hybrid (the fine-tune's root, the
  500M model's leaves) against the 500M model alone, both at depth 2, which is what
  the bot runs. 400 starting positions, each twice with colours swapped: the first
  200 are the 200 of every match before (same shard, same seed, the same shuffle), so
  they pair with results/27 and results/29. Both sides play src/root_ranking.py's
  one-pass player, the search's moves in two network calls per move.
- Card 0: the light fine-tune again (rank weight 0.3, same start, data, steps and
  schedule as results/26) with one change: on every action-value batch the student
  also pays KL(500M || student) over the 64 buckets, the frozen 500M model as the
  teacher. If the leaves are what the ranking loss damages, this should keep them,
  and with them the gain from search. Then the same evaluations as results/32: the
  10,000 puzzles at depth 1, move agreement on the test shard, and the root/leaf pass
  against the 500M model.

The match reads its own copy of the test shard: the packing cell deletes every .bag
it finds in DATA, and the agreement pass downloads the shard again later. Deadlines
come from the session clock as in build_root_ranking.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, data_cell, setup_cell  # noqa: E402
from build_puzzle_finetune import puzzle_data_cell  # noqa: E402

BASE_KERNEL = "nicolangileri/gpt-from-scratch-chess-500m-s3"            # the 500M model
W03_KERNEL = "nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03"     # its w0.3 fine-tune
DIR = "kaggle_chess_distill_hybrid"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-distill-hybrid"
SHARDS = 3                    # as the w0.3 fine-tune (results/26)
STEPS, BATCH, RANK_BATCH, LR = 12000, 1024, 64, 1e-4
RANK_WEIGHT, DISTILL_WEIGHT = 0.3, 1.0
MATCH_POSITIONS = 400
TEST_BAG = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
EVAL_RESERVE_H = 2.2          # puzzles ~0.2h, agreement ~0.5h, root/leaf pass ~1.4h
MATCH_EARLY_H = 0.10          # a position started before its deadline still plays
HARD_STOP_AFTER_H = 0.05


def start_cell(end_h: float) -> str:
    """Locate the checkpoints and start the match on card 1."""
    return f'''\
import glob, os, subprocess, sys, threading, time, urllib.request
END = T0 + {end_h} * 3600
hours_left = lambda: (END - time.time()) / 3600
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
MATCH_DIR = DATA + "_match"                   # outside DATA: the packing deletes *.bag
os.makedirs(DATA, exist_ok=True)
os.makedirs(MATCH_DIR, exist_ok=True)
urllib.request.urlretrieve("{TEST_BAG}", MATCH_DIR + "/test.bag")
assert os.path.getsize(MATCH_DIR + "/test.bag") == 148_297_469, "not the test shard"

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

MATCH_JOB = subprocess.Popen(
    [sys.executable, "-m", "src.match", "--a", W03, "--a_leaf", BASE500, "--b", BASE500,
     "--depth", "2", "--onepass", "--positions", "{MATCH_POSITIONS}",
     "--bag", MATCH_DIR + "/test.bag", "--device", "cuda", "--threads", "2", "--seed", "0",
     "--hours", str(hours_left() - {MATCH_EARLY_H}), "--out", "match_hybrid_vs_500M_d2.json"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    env={{**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}})
MATCH_READER = threading.Thread(target=stream, args=("match", MATCH_JOB), daemon=True)
MATCH_READER.start()
print(f"[session] match started {{(time.time() - T0) / 3600:.2f}}h in, jobs end at {end_h}h",
      flush=True)
'''


def finetune_cell() -> str:
    """build_puzzle_finetune.finetune_cell's command for the w0.3 run, plus the teacher.
    "cuda" is card 0: the match was started with only card 1 visible."""
    return f'''\
import sys, time
INIT = BASE500
budget = END - {EVAL_RESERVE_H} * 3600 - time.time()
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--data_dir", DATA + "/chess", "--out_dir", "checkpoints",
            "--puzzle_dir", DATA + "/puzzles", "--rank_batch", "{RANK_BATCH}",
            "--rank_weight", "{RANK_WEIGHT}",
            "--distill_from", BASE500, "--distill_weight", "{DISTILL_WEIGHT}",
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
D = "chess_500M_w03_distill.pt"
shutil.copy("checkpoints/chess_best.pt", D)       # a name that says what it is
results = {{"puzzles": puzzles(D, DATA + "/puzzles.csv", 10000, "cuda",
                              out="puzzles_500M_w03_distill.json")}}
json.dump(results, open("results_w03_distill.json", "w"), indent=1, default=float)
urllib.request.urlretrieve("{TEST_BAG}", DATA + "/test.bag")
results["agreement"] = agreement(D, DATA + "/test.bag", None, "cuda")
json.dump(results, open("results_w03_distill.json", "w"), indent=1, default=float)
print(f"[session] root/leaf pass from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
root_ranking({{"500M_w03d": D, "500M": BASE500}}, DATA + "/puzzles.csv", 10000, "cuda",
             hours=hours_left(), out="root_ranking_500M_w03_distill.json")
'''


def wait_cell() -> str:
    return f'''\
import subprocess, time
try:
    code = MATCH_JOB.wait(timeout=max(1, END + {HARD_STOP_AFTER_H} * 3600 - time.time()))
except subprocess.TimeoutExpired:
    MATCH_JOB.kill()
    code = "killed at the hard stop"
MATCH_READER.join(timeout=10)
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: match {{code}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 7.5 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; the fine-tune "
                         "needs ~1.4h of packing, ~5h of training and the evaluations")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "distill_hybrid_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), data_cell(SHARDS),
         puzzle_data_cell(None, 20_000), finetune_cell(), eval_cell(), wait_cell()]),
        indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "distill_hybrid_kernel.ipynb", "dataset_sources": [],
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
