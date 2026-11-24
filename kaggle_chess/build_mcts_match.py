"""One GPU session: the network-guided tree search (src/mcts.py) against the bot.

    python kaggle_chess/build_mcts_match.py --quota_h 6.5
    kaggle kernels push -p kaggle_chess_mcts_match/

The bot plays alpha-beta at depth 2 on the 1B hybrid (the 1B model's w0.3 fine-tune at
the root, the 1B model at the leaves); depth 2 is worth +165 Elo over no search
(results/38), a third ply at fixed depth +15 (results/34). Does a tree search that
spends its calls where the network is unsure do better with the same networks?

Both sides get about the same network work per move. Depth 2 in one pass evaluates
the root's moves and every child's moves, ~30 + ~900 (position, move) pairs; MCTS
with 32 simulations makes 33 calls of ~30 pairs each. Fewer pairs per call, so
more seconds per move on a GPU: both are recorded.

1. Tuning, on the first 2,000 of the 10,000 test puzzles, MCTS with 32 simulations and
   the bot's networks at c_puct 0.03, 0.1, 0.3 and 1.0 (prior temperature 0.05), and
   depth 2 on the same puzzles as the reference. The c played in the match is the one
   with the most puzzles solved, the smaller on a tie. Puzzles only pick c; the games
   decide.
2. The match: MCTS (that c, 32 simulations) against the bot, 800 starting positions
   played twice with colours swapped, seed 0 (those of results/33-38), split over both
   cards by `--slice`. Decided before the games: the bot changes to MCTS only if the
   95% interval, clustering the two games of each position, excludes zero in MCTS's
   favour.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, SOURCES, _notebook  # noqa: E402

ONE_B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-s5"             # chess_1B.pt
W1B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-w03-match"        # chess_1B_w03.pt
DIR = "kaggle_chess_mcts_match"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-mcts-match"
MCTS_SOURCES = SOURCES                                 # src/mcts.py is in SOURCES
BASE = "https://storage.googleapis.com/searchless_chess/data/"
TEST_BAG = BASE + "test/action_value_data.bag"
POSITIONS = 800
SIMS = 32
TAU = 0.05
C_GRID = (0.03, 0.1, 0.3, 1.0)
TUNE_PUZZLES = 2000
END_BEFORE_QUOTA_H = 0.08
MATCH_EARLY_H = 0.10
HARD_STOP_AFTER_H = 0.05


def setup_cell() -> str:
    blob = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in MCTS_SOURCES}
    return f'''\
import base64, os, time
T0 = time.time()                      # the session clock starts here
os.system("pip install -q chess")
_FILES = {json.dumps(blob)}
for path, b64 in _FILES.items():
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("wrote", list(_FILES))
'''


def start_cell(end_h: float) -> str:
    return f'''\
import glob, json, os, subprocess, sys, threading, time, urllib.request
END = T0 + {end_h} * 3600
hours_left = lambda: (END - time.time()) / 3600
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("{TEST_BAG}", DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard"
urllib.request.urlretrieve("{BASE}puzzles.csv", DATA + "/puzzles.csv")

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
W1B = one("/kaggle/input/**/{W1B_KERNEL.split("/")[1]}/**/chess_1B_w03.pt")
ONE_B = one("/kaggle/input/**/{ONE_B_KERNEL.split("/")[1]}/**/chess_best.pt")
print("[session] root", W1B, "| leaves", ONE_B, flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

def launch(args, card, name):
    proc = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(card),
                                  "PYTHONUNBUFFERED": "1"}})
    reader = threading.Thread(target=stream, args=(name, proc), daemon=True)
    reader.start()
    return proc, reader

def finish(proc, reader):
    try:
        code = proc.wait(timeout=max(1, END + {HARD_STOP_AFTER_H} * 3600 - time.time()))
    except subprocess.TimeoutExpired:
        proc.kill()
        code = "killed at the hard stop"
    reader.join(timeout=10)
    return code
'''


def tune_cell() -> str:
    grid = json.dumps(list(C_GRID))
    return f'''\
def tune_job(c):
    out = "tune_depth2.json" if c is None else f"tune_mcts_c{{c}}.json"
    args = ["src.mcts", "--ckpt", W1B, "--leaf_ckpt", ONE_B, "--sims", "{SIMS}",
            "--tau", "{TAU}", "--puzzles", DATA + "/puzzles.csv", "--num", "{TUNE_PUZZLES}",
            "--device", "cuda", "--out", out]
    return args + (["--depth2"] if c is None else ["--c_puct", str(c)]), out

GRID = {grid}
QUEUES = [GRID[0::2] + [None], GRID[1::2]]       # card 0 also runs the depth-2 reference

def run_queue(card, queue, codes):
    for c in queue:
        args, out = tune_job(c)
        codes[out] = finish(*launch(args, card, out.split(".")[0]))

print(f"[session] tuning from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
tune_codes = {{}}
threads = [threading.Thread(target=run_queue, args=(card, q, tune_codes)) for card, q in enumerate(QUEUES)]
for t in threads: t.start()
for t in threads: t.join()
tuned = {{c: json.load(open(tune_job(c)[1])) for c in GRID}}
C = max(GRID, key=lambda c: (tuned[c]["solved"], -c))
ref = json.load(open("tune_depth2.json"))
summary = {{"grid": {{str(c): {{k: tuned[c][k] for k in ("solved", "attempted", "seconds_per_move")}}
                     for c in GRID}},
           "depth2": {{k: ref[k] for k in ("solved", "attempted", "seconds_per_move")}},
           "chosen_c": C, "codes": tune_codes}}
json.dump(summary, open("tune_summary.json", "w"), indent=1)
print(f"[session] tuned at {{(time.time() - T0) / 3600:.2f}}h: {{summary}}", flush=True)
'''


def match_cell() -> str:
    return f'''\
hours = hours_left() - {MATCH_EARLY_H}
print(f"[session] match from {{(time.time() - T0) / 3600:.2f}}h, c {{C}}, {{hours:.2f}}h per card",
      flush=True)
cards = []
for card in (0, 1):
    out = f"match_mcts{SIMS}_vs_bot_part{{card}}.json"
    args = ["src.match", "--a", W1B, "--a_leaf", ONE_B, "--a_sims", "{SIMS}",
            "--c_puct", str(C), "--tau", "{TAU}",
            "--b", W1B, "--b_leaf", ONE_B, "--depth", "2", "--onepass",
            "--positions", "{POSITIONS}", "--slice", f"{{card}}/2",
            "--bag", DATA + "/test.bag", "--device", "cuda", "--threads", "2",
            "--seed", "0", "--hours", str(hours), "--out", out]
    cards.append(launch(args, card, out.split(".")[0]))
codes = [finish(*p) for p in cards]
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 5 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; tuning takes "
                         "~1h and the match ~4h")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "mcts_match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), tune_cell(), match_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "mcts_match_kernel.ipynb", "dataset_sources": [],
                 "kernel_sources": [W1B_KERNEL, ONE_B_KERNEL]})
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
