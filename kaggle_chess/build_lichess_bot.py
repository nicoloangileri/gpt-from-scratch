"""The Lichess bot in a Kaggle CPU session: the 1B hybrid, online for up to 11 hours.

    python kaggle_chess/build_lichess_bot.py
    kaggle kernels push -p kaggle_chess_lichess_bot/

The bot plays what results/37, 39, 41 and 46 chose: the 1B model's w0.3 puzzle
fine-tune at the root, the 1B model below it, and the tree search of src/mcts.py
(128 simulations with three minutes or more on the clock, 64 below, c 0.3) instead of
depth 2, seeking games against other bots (`--seek`). A CPU session, which uses no GPU
quota: on a Kaggle CPU 64 simulations take 3.8 s per move at the median and 4.8 s at
worst (results/42), depth 2 took 2.9 and 5.0 (results/40); 128 cost exactly twice 64 on
a T4 (results/46), so about 7.7 s here.

CLOCKS sets the session's speed. results/43 played rapid (10+3, 1781 to 1855); the
paper's searchless 9M model is rated 2054 in *blitz* against bots, so this session plays
blitz only: it offers 5+3, then 3+2, and declines any other speed. Replaying results/43's
own move times, 5+3 keeps 98.5% of moves at the full 64 simulations; 3+2 halves them
under 90 s for about half the moves and never reaches the bare network.

The token never touches this repository, the notebook or its log. It comes from a
Kaggle secret named LICHESS_TOKEN, which the account owner adds and attaches to this
notebook in the Kaggle editor (Add-ons -> Secrets); the notebook hands it to the bot
only through the environment of the bot's own process, the one place src/lichess_bot.py
reads it from. Without the secret the notebook says so and ends.

Stopping: Kaggle kills a session at 12 hours, and a bot killed mid-game loses that game
on time. At STOP_H the notebook sends the bot SIGTERM, which stops new games and lets
the one on the board finish; if it is still on at LEAVE_H, a second SIGTERM leaves at
once.
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
DIR = "kaggle_chess_lichess_bot"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-lichess-bot"
BOT_SOURCES = SOURCES + ["src/lichess_bot.py"]
SECRET = "LICHESS_TOKEN"
CLOCKS = "5+3,3+2"            # blitz only; None: rapid first, as in results/38 and 43
STOP_H = 11.0                 # no new games after this
LEAVE_H = 11.75               # leave even mid-game: Kaggle kills at 12h


def setup_cell() -> str:
    blob = {p: base64.b64encode((ROOT / p).read_bytes()).decode() for p in BOT_SOURCES}
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


def models_cell() -> str:
    return f'''\
import glob

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
ROOT_CKPT = one("/kaggle/input/**/{W1B_KERNEL.split("/")[1]}/**/chess_1B_w03.pt")
LEAF_CKPT = one("/kaggle/input/**/{ONE_B_KERNEL.split("/")[1]}/**/chess_best.pt")
print("[session] root", ROOT_CKPT, "| leaves", LEAF_CKPT, flush=True)
'''


def secret_cell() -> str:
    """Read the token. Nothing about it is printed: not the value, not its length, not
    the error text of a failed read, which could quote it."""
    return f'''\
try:
    from kaggle_secrets import UserSecretsClient
    _TOKEN = (UserSecretsClient().get_secret("{SECRET}") or "").strip() or None
except Exception as exc:                      # not attached, or not a Kaggle session
    print(f"[session] secret read failed: {{type(exc).__name__}}", flush=True)
    _TOKEN = None
if _TOKEN is None:
    print("[session] no {SECRET} secret on this notebook: add it in the editor under "
          "Add-ons -> Secrets, attach it, and run again. Not starting the bot.", flush=True)
'''


def bot_cell() -> str:
    clocks = f', "--clocks", "{CLOCKS}"' if CLOCKS else ""
    return f'''\
import os, signal, subprocess, sys, threading, time

def run_bot():
    env = {{**os.environ, "{SECRET}": _TOKEN, "PYTHONUNBUFFERED": "1"}}
    check = subprocess.run([sys.executable, "-m", "src.lichess_bot", "--check"],
                           env=env, capture_output=True, text=True)
    print(check.stdout + check.stderr, flush=True)
    if check.returncode:
        print("[session] the token check failed: not starting the bot", flush=True)
        return
    bot = subprocess.Popen(
        [sys.executable, "-m", "src.lichess_bot", "--ckpt", ROOT_CKPT, "--leaf_ckpt", LEAF_CKPT,
         "--sims", "128", "--c_puct", "0.3", "--device", "cpu", "--seek"{clocks}],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def stream():
        for line in bot.stdout:
            print(line, end="", flush=True)
    reader = threading.Thread(target=stream, daemon=True)
    reader.start()
    print(f"[session] bot started {{(time.time() - T0) / 3600:.2f}}h in", flush=True)
    for deadline, what in ((T0 + {STOP_H} * 3600, "stop: no new games, finish this one"),
                           (T0 + {LEAVE_H} * 3600, "leave now")):
        try:
            bot.wait(timeout=max(1, deadline - time.time()))
            break
        except subprocess.TimeoutExpired:
            print(f"[session] {{(time.time() - T0) / 3600:.2f}}h: {{what}}", flush=True)
            bot.send_signal(signal.SIGTERM)
    try:
        bot.wait(timeout=120)
    except subprocess.TimeoutExpired:
        bot.kill()
    reader.join(timeout=10)
    print(f"[session] bot ended at {{(time.time() - T0) / 3600:.2f}}h, code {{bot.returncode}}",
          flush=True)

if _TOKEN is not None:
    run_bot()
'''


def build(out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "lichess_bot_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), models_cell(), secret_cell(), bot_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "lichess_bot_kernel.ipynb", "enable_gpu": False,
                 "enable_internet": True, "dataset_sources": [],
                 "kernel_sources": [W1B_KERNEL, ONE_B_KERNEL]})
    meta.pop("machine_shape", None)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


if __name__ == "__main__":
    print("wrote", build(ROOT / DIR).name + "/")
