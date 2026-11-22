"""One GPU session, no training: what the bot's search and its fine-tuned root are worth
on the 1B model, in direct matches.

    python kaggle_chess/build_1b_search_match.py --quota_h 7
    kaggle kernels push -p kaggle_chess_1b_search_match/

results/37 measured the 1B model against the 500M bot twice: the 1B alone (root and
leaves, depth 2) +45 Elo, the 1B hybrid (its w0.3 puzzle fine-tune at the root, the 1B
model at the leaves, depth 2) +51. The 1B hybrid became the bot; the two were never
played against each other, and the 6 Elo between them is the difference of two
estimates with ~14 Elo of noise each.

Two matches, A always the bot's configuration (the 1B hybrid, depth 2, one-pass player):

1. Against the 1B model alone at depth 2: is the fine-tuned root worth anything on top
   of the 1B model?
2. Against the 1B model at depth 1, the searchless player of the paper: what the
   depth-2 search is worth.

Each match: 800 starting positions played twice with colours swapped, seed 0 (the
positions of results/33, 35 and 37), split over both cards by `--slice`, so a cut at
the deadline leaves each card an even sample of the list. Decided before the games: the
bot changes configuration only if the 95% interval, clustering the two games of each
position, excludes zero against the bot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook, setup_cell  # noqa: E402

ONE_B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-s5"             # chess_1B.pt
W1B_KERNEL = "nicolangileri/gpt-from-scratch-chess-1b-w03-match"        # chess_1B_w03.pt
DIR = "kaggle_chess_1b_search_match"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-1b-search-match"
POSITIONS = 800
TEST_BAG = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
END_BEFORE_QUOTA_H = 0.08     # outside T0..end: start-up and saving
FIRST_MATCH_H = 3.0           # 400 positions per card took 2.5-2.6h in results/37
MATCH_EARLY_H = 0.10          # a position started before its deadline still plays
HARD_STOP_AFTER_H = 0.05
MATCHES = (                   # name, B's root, B's leaves, B's depth
    ("1B_hybrid_vs_1B_d2", "ONE_B", "ONE_B", 2),
    ("1B_hybrid_vs_1B_d1", "ONE_B", "ONE_B", 1),
)


def start_cell(end_h: float) -> str:
    return f'''\
import glob, os, subprocess, sys, threading, time, urllib.request
END = T0 + {end_h} * 3600
hours_left = lambda: (END - time.time()) / 3600
DATA = "/kaggle/temp/data" if os.path.isdir("/kaggle/temp") else "/tmp/data"
os.makedirs(DATA, exist_ok=True)
urllib.request.urlretrieve("{TEST_BAG}", DATA + "/test.bag")
assert os.path.getsize(DATA + "/test.bag") == 148_297_469, "not the test shard"

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert len(hits) == 1, f"{{pattern}}: {{hits}}"
    return hits[0]
W1B = one("/kaggle/input/**/{W1B_KERNEL.split("/")[1]}/**/chess_1B_w03.pt")
ONE_B = one("/kaggle/input/**/{ONE_B_KERNEL.split("/")[1]}/**/chess_best.pt")
print("[session] 1B hybrid root", W1B, "| 1B", ONE_B, flush=True)

def stream(name, proc):
    for line in proc.stdout:
        print(f"[{{name}}] {{line}}", end="", flush=True)

def match(b, b_leaf, b_depth, card, hours, out, slice_):
    """The bot (the 1B hybrid, depth 2) against b, on one card."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.match", "--a", W1B, "--a_leaf", ONE_B,
         "--b", b, "--b_leaf", b_leaf, "--depth", "2", "--a_depth", "2",
         "--b_depth", str(b_depth), "--onepass",
         "--positions", "{POSITIONS}", "--slice", slice_,
         "--bag", DATA + "/test.bag", "--device", "cuda", "--threads", "2",
         "--seed", "0", "--hours", str(hours), "--out", out],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONUNBUFFERED": "1"}})
    reader = threading.Thread(target=stream, args=(out.split(".")[0], proc), daemon=True)
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


def matches_cell() -> str:
    names = json.dumps([m[0] for m in MATCHES])
    return f'''\
codes = {{}}
SPECS = [{", ".join(f'("{n}", {r}, {lf}, {d})' for n, r, lf, d in MATCHES)}]
assert [s[0] for s in SPECS] == {names}
for k, (name, b, b_leaf, b_depth) in enumerate(SPECS):
    # the first match may use at most FIRST_MATCH_H, the last one what is left
    hours = (min({FIRST_MATCH_H}, hours_left()) if k == 0 else hours_left()) - {MATCH_EARLY_H}
    if hours <= 0:
        print(f"[session] no time left for {{name}}", flush=True)
        break
    print(f"[session] {{name}} from {{(time.time() - T0) / 3600:.2f}}h, "
          f"{{hours:.2f}}h per card", flush=True)
    cards = [match(b, b_leaf, b_depth, c, hours, f"match_{{name}}_part{{c}}.json", f"{{c}}/2")
             for c in (0, 1)]
    codes[name] = [finish(*p) for p in cards]
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 5 <= end_h <= 11:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; the first "
                         f"match needs {FIRST_MATCH_H}h and the second ~2h")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "search_match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), matches_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "search_match_kernel.ipynb", "dataset_sources": [],
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
