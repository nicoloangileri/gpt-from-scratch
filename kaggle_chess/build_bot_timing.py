"""Seconds per move of the bot's two searches on a Kaggle CPU, where the bot plays.

    python kaggle_chess/build_bot_timing.py
    kaggle kernels push -p kaggle_chess_bot_timing/

results/39 timed MCTS (32 simulations) on a T4: 0.28 s per move against 0.05 for the
bot's depth 2 in one pass. Online the bot runs src/lichess_bot.py on a Kaggle CPU
session, where depth 2 is one network call per node rather than two batched ones, so
the GPU numbers do not say whether MCTS fits the clock. This session plays the bot's
own `pick` -- the same code, the same two networks, the CPU, the threads torch picks --
on the first POSITIONS of the 800 match positions (seed 0), alternating the two
settings position by position so a slow stretch of the machine hits both alike.

results/41 then put 64 simulations 56 Elo above 32; the second session times those two.

A CPU kernel: no GPU quota. No token, no Lichess: `pick` needs neither.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, SOURCES, _notebook  # noqa: E402

ONE_B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-s5"             # chess_1B.pt
W1B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-w03-match"        # chess_1B_w03.pt
DIR = "kaggle_chess_bot_timing"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-bot-timing"
TIMING_SOURCES = SOURCES + ["src/lichess_bot.py"]
TEST_BAG = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
POSITIONS = 100
SETTINGS = {"mcts32": 32, "mcts64": 64}   # results/40 was {"depth2": None, "mcts32": 32}
C_PUCT = 0.3
TAU = 0.05
HOURS = 2.0                   # stop between positions after this long


def setup_cell() -> str:
    blob = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in TIMING_SOURCES}
    return f'''\
import base64, os, time
T0 = time.time()
os.system("pip install -q chess")
_FILES = {json.dumps(blob)}
for path, b64 in _FILES.items():
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(base64.b64decode(b64))
print("wrote", list(_FILES))
'''


def timing_cell() -> str:
    return f'''\
import glob, json, os, time, urllib.request
import chess, numpy as np, torch
from src.lichess_bot import LichessBot
from src.match import balanced_positions
from src.puzzles_eval import load_model

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
ROOT_CKPT = one("/kaggle/input/**/{W1B_KERNEL.split("/")[1]}/**/chess_1B_w03.pt")
LEAF_CKPT = one("/kaggle/input/**/{ONE_B_KERNEL.split("/")[1]}/**/chess_best.pt")
os.makedirs("/kaggle/temp", exist_ok=True)
urllib.request.urlretrieve("{TEST_BAG}", "/kaggle/temp/test.bag")
fens = balanced_positions("/kaggle/temp/test.bag", 800, 0)[:{POSITIONS}]

bot = LichessBot.__new__(LichessBot)          # pick() only: no token, no session
bot.model, bot.centres, _ = load_model(ROOT_CKPT, "cpu")
bot.leaf_model, bot.leaf_centres, _ = load_model(LEAF_CKPT, "cpu")
bot.device, bot.depth = "cpu", 2
SETTINGS = {json.dumps(SETTINGS).replace("null", "None")}
bot.c_puct, bot.tau = {C_PUCT}, {TAU}
print(f"[timing] {{os.cpu_count()}} CPUs, torch threads {{torch.get_num_threads()}}", flush=True)

times = {{k: [] for k in SETTINGS}}
moves = {{k: [] for k in SETTINGS}}
for i, fen in enumerate(fens):
    if time.time() - T0 > {HOURS} * 3600:
        print(f"[timing] stopped after {{i}} positions", flush=True)
        break
    order = list(SETTINGS) if i % 2 == 0 else list(SETTINGS)[::-1]
    for name in order:
        bot.sims = SETTINGS[name]
        t = time.perf_counter()
        moves[name].append(bot.pick(chess.Board(fen), 600_000))
        times[name].append(time.perf_counter() - t)
    if (i + 1) % 10 == 0:
        print(f"[timing] {{i + 1}} positions: " + ", ".join(
            f"{{k}} median {{np.median(v):.2f}} s" for k, v in times.items()), flush=True)

def stats(t):
    t = np.asarray(t)
    return {{"moves": len(t), "mean_s": float(t.mean()), "median_s": float(np.median(t)),
            "p90_s": float(np.quantile(t, 0.9)), "max_s": float(t.max())}}
out = {{"cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "settings": SETTINGS, "c_puct": {C_PUCT}, "tau": {TAU},
        "same_move": float(np.mean([a == b for a, b in zip(*moves.values())])),
        **{{k: stats(v) for k, v in times.items()}}, "times": times}}
json.dump(out, open("bot_timing.json", "w"), indent=1)
print("[timing]", {{k: v for k, v in out.items() if k != "times"}}, flush=True)
'''


def build(out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "bot_timing_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), timing_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "bot_timing_kernel.ipynb", "enable_gpu": False,
                 "enable_internet": True, "dataset_sources": [],
                 "kernel_sources": [W1B_KERNEL, ONE_B_KERNEL]})
    meta.pop("machine_shape", None)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


if __name__ == "__main__":
    print("wrote", build(ROOT / DIR).name + "/")
