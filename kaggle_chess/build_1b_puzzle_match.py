"""One GPU session on the 1B model: its puzzle fine-tune, and two matches against the bot.

    python kaggle_chess/build_1b_puzzle_match.py --quota_h 11
    kaggle kernels push -p kaggle_chess_1b_puzzle_match/

The bot plays the hybrid: the 500M model's w0.3 puzzle fine-tune at the root, the 500M
model at the leaves, depth 2 (results/33, +26 Elo against the 500M model alone). The 1B
model (results/36) solves 68.0% of the puzzles against the 500M model's 64.8%. Does
it also make a better bot?

- Card 0: the w0.3 fine-tune again, the same command as results/26 with the 1B model as
  the start: rank weight 0.3, 3 shards, 12,000 steps, lr 1e-4 -> 1e-5. Then the 10,000
  puzzles at depth 1 and move agreement on the test shard, as for every checkpoint.
- Card 1, from the start: the 1B model alone (root and leaves) against the bot. It says
  whether the bigger run is worth anything in games without a fine-tune.
- Both cards, once the fine-tune is evaluated: the 1B hybrid (its fine-tune at the root,
  the 1B model at the leaves) against the bot. Card 0 starts as soon as the evaluation
  is done, card 1 when its first match is; each takes every second position of the
  list, so a cut at the deadline leaves an even sample of it.

All matches: depth 2 with the one-pass player, 800 starting positions each played twice
with colours swapped, seed 0, so the first 400 are those of results/33 and results/35.
Decided before the games: the bot changes configuration only if the 95% interval,
clustering the two games of each position, excludes zero in the challenger's favour.

The matches read their own copy of the test shard: the packing cell deletes every .bag
it finds in DATA, and the agreement pass downloads the shard again later.
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
ONE_B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-s5"             # the 1B model
DIR = "kaggle_chess_1b_puzzle_match"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-1b-w03-match"
SHARDS = 3                    # as the w0.3 fine-tune (results/26)
STEPS, BATCH, RANK_BATCH, LR, RANK_WEIGHT = 12000, 1024, 64, 1e-4, 0.3
POSITIONS = 800
TEST_BAG = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
MAIN_MATCH_H = 3.0            # 400 positions on one T4 at ~26 s each (results/35)
EVAL_H = 0.8                  # puzzles ~0.2h, agreement ~0.5h
MATCH_EARLY_H = 0.10          # a position started before its deadline still plays
HARD_STOP_AFTER_H = 0.05


def start_cell(end_h: float) -> str:
    """Locate the checkpoints and start the first match on card 1."""
    return f'''\
import glob, os, subprocess, sys, threading, time, urllib.request
END = T0 + {end_h} * 3600
MAIN_START = END - {MAIN_MATCH_H} * 3600     # the latest the second match may start
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
ONE_B = one("/kaggle/input/**/{ONE_B_KERNEL.split("/")[1]}/**/chess_best.pt")
print("[session] 500M", BASE500, "| w0.3", W03, "| 1B", ONE_B, flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

def match(a, a_leaf, card, hours, out, slice_="0/1"):
    """a (with a_leaf at the leaves) against the bot's hybrid, on one card."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.match", "--a", a, "--a_leaf", a_leaf,
         "--b", W03, "--b_leaf", BASE500, "--depth", "2", "--onepass",
         "--positions", "{POSITIONS}", "--slice", slice_,
         "--bag", MATCH_DIR + "/test.bag", "--device", "cuda", "--threads", "2",
         "--seed", "0", "--hours", str(hours), "--out", out],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONUNBUFFERED": "1"}})
    reader = threading.Thread(target=stream, args=(out.split(".")[0], proc), daemon=True)
    reader.start()
    return proc, reader

PLAIN = match(ONE_B, ONE_B, 1, (MAIN_START - time.time()) / 3600 - {MATCH_EARLY_H},
              "match_1B_vs_hybrid_d2.json")
print(f"[session] first match started {{(time.time() - T0) / 3600:.2f}}h in, "
      f"jobs end at {end_h}h", flush=True)
'''


def finetune_cell() -> str:
    """The w0.3 command of results/26 with the 1B model as the start. "cuda" is card 0:
    the first match was started with only card 1 visible."""
    return f'''\
import sys, time
INIT = ONE_B
budget = MAIN_START - {EVAL_H} * 3600 - time.time()
print(f"[session] training budget {{budget / 3600:.2f}}h", flush=True)
sys.argv = ["train_chess", "--config", "configs/chess.yaml", "--device", "cuda",
            "--data_dir", DATA + "/chess", "--out_dir", "checkpoints",
            "--puzzle_dir", DATA + "/puzzles", "--rank_batch", "{RANK_BATCH}",
            "--rank_weight", "{RANK_WEIGHT}",
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
W1B = "chess_1B_w03.pt"
shutil.copy("checkpoints/chess_best.pt", W1B)     # a name that says what it is
results = {{"puzzles": puzzles(W1B, DATA + "/puzzles.csv", 10000, "cuda",
                              out="puzzles_1B_w03.json")}}
json.dump(results, open("results_1B_w03.json", "w"), indent=1, default=float)
urllib.request.urlretrieve("{TEST_BAG}", DATA + "/test.bag")
results["agreement"] = agreement(W1B, DATA + "/test.bag", None, "cuda")
json.dump(results, open("results_1B_w03.json", "w"), indent=1, default=float)
'''


def main_match_cell() -> str:
    return f'''\
import subprocess, time, torch
torch.cuda.empty_cache()      # what the fine-tune left cached on card 0

def finish(proc, reader):
    try:
        code = proc.wait(timeout=max(1, END + {HARD_STOP_AFTER_H} * 3600 - time.time()))
    except subprocess.TimeoutExpired:
        proc.kill()
        code = "killed at the hard stop"
    reader.join(timeout=10)
    return code

hours = lambda: hours_left() - {MATCH_EARLY_H}
print(f"[session] second match from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
card0 = match(W1B, ONE_B, 0, hours(), "match_1B_hybrid_vs_hybrid_part0.json", "0/2")
codes = {{"1B_vs_hybrid": finish(*PLAIN)}}
print(f"[session] card 1 free at {{(time.time() - T0) / 3600:.2f}}h", flush=True)
card1 = match(W1B, ONE_B, 1, hours(), "match_1B_hybrid_vs_hybrid_part1.json", "1/2")
codes["part0"], codes["part1"] = finish(*card0), finish(*card1)
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 9 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; the session "
                         "needs ~1.5h of packing, ~3h of fine-tune, the evaluations and "
                         f"{MAIN_MATCH_H}h of the second match")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "puzzle_match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), data_cell(SHARDS),
         puzzle_data_cell(None, 20_000), finetune_cell(), eval_cell(), main_match_cell()]),
        indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "puzzle_match_kernel.ipynb", "dataset_sources": [],
                 "kernel_sources": [BASE_KERNEL, W03_KERNEL, ONE_B_KERNEL]})
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota_h", type=float, required=True,
                    help="GPU hours this session may use; check `kaggle quota` and keep a margin")
    a = ap.parse_args()
    out = build(a.quota_h, ROOT / DIR)
    print(f"wrote {out.name}/: jobs end {a.quota_h - END_BEFORE_QUOTA_H:.2f}h after the "
          "session starts")


if __name__ == "__main__":
    main()
