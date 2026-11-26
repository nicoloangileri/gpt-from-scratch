"""One GPU session: are c 0.3 and prior temperature 0.05 still right at 64 simulations?

    python kaggle_chess/build_mcts_tune64.py --quota_h 10.4
    kaggle kernels push -p kaggle_chess_mcts_tune64/

The bot plays the tree search of src/mcts.py with 64 simulations (results/41), at the
c_puct 0.3 that results/39 picked on puzzles with 32 simulations, and at a prior
temperature of 0.05 that was never tuned. With twice the simulations a larger c may
pay: the search can afford to look at more moves.

1. Tuning, on the first 2,000 of the 10,000 test puzzles (those of results/39), the
   bot's networks with 64 simulations at every (c, temperature) of GRID, the bot's own
   setting among them. The challenger is the setting other than the bot's with the
   most puzzles solved; on a tie, the one nearer the bot's (smaller c, then smaller
   temperature). Puzzles only pick the challenger; the games decide. The challenger is
   played even if the bot's setting solves more: a session's quota is spent either way,
   and the games measure what the puzzles cannot.
2. The match: the challenger against the bot's setting, both at 64 simulations with the
   bot's networks, on the 800 positions of results/33-41 (seed 0) played twice with
   colours swapped, split over both cards by `--slice`; the deadline cuts the list
   cleanly between positions. Decided before the games: the bot changes setting only if
   the 95% interval, clustering the two games of each position, excludes zero in the
   challenger's favour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook  # noqa: E402
from build_mcts_match import (ONE_B_KERNEL, POSITIONS, TUNE_PUZZLES, W1B_KERNEL,  # noqa: E402
                              setup_cell, start_cell)

DIR = "kaggle_chess_mcts_tune64"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-mcts-tune64"
SIMS = 64
BOT = (0.3, 0.05)                                     # the bot's (c, temperature)
GRID = [(0.3, 0.05), (0.6, 0.05), (1.0, 0.05), (0.3, 0.1), (0.6, 0.1), (1.0, 0.1)]
END_BEFORE_QUOTA_H = 0.08
MATCH_EARLY_H = 0.10


def tune_cell() -> str:
    return f'''\
GRID = {json.dumps(GRID)}
BOT = {json.dumps(list(BOT))}

def tune_job(c, tau):
    out = f"tune64_c{{c}}_tau{{tau}}.json"
    return ["src.mcts", "--ckpt", W1B, "--leaf_ckpt", ONE_B, "--sims", "{SIMS}",
            "--c_puct", str(c), "--tau", str(tau), "--puzzles", DATA + "/puzzles.csv",
            "--num", "{TUNE_PUZZLES}", "--device", "cuda", "--out", out], out

QUEUES = [GRID[0::2], GRID[1::2]]

def run_queue(card, queue, codes):
    for c, tau in queue:
        args, out = tune_job(c, tau)
        codes[out] = finish(*launch(args, card, out[:-5]))

print(f"[session] tuning from {{(time.time() - T0) / 3600:.2f}}h", flush=True)
tune_codes = {{}}
threads = [threading.Thread(target=run_queue, args=(card, q, tune_codes))
           for card, q in enumerate(QUEUES)]
for t in threads: t.start()
for t in threads: t.join()
tuned = {{tuple(s): json.load(open(tune_job(*s)[1])) for s in GRID}}
others = [tuple(s) for s in GRID if list(s) != BOT]
CHALLENGER = max(others, key=lambda s: (tuned[s]["solved"], -s[0], -s[1]))
summary = {{"sims": {SIMS},
           "grid": {{f"c{{c}}_tau{{tau}}": {{k: tuned[(c, tau)][k]
                                          for k in ("solved", "attempted", "seconds_per_move")}}
                    for c, tau in tuned}},
           "bot": BOT, "challenger": list(CHALLENGER), "codes": tune_codes}}
json.dump(summary, open("tune64_summary.json", "w"), indent=1)
print(f"[session] tuned at {{(time.time() - T0) / 3600:.2f}}h: {{summary}}", flush=True)
'''


def match_cell() -> str:
    return f'''\
hours = hours_left() - {MATCH_EARLY_H}
c, tau = CHALLENGER
print(f"[session] match from {{(time.time() - T0) / 3600:.2f}}h: c {{c}} tau {{tau}} against "
      f"the bot's c {{BOT[0]}} tau {{BOT[1]}}, {{hours:.2f}}h per card", flush=True)
cards = []
for card in (0, 1):
    out = f"match_mcts64_c{{c}}_tau{{tau}}_vs_bot_part{{card}}.json"
    args = ["src.match", "--a", W1B, "--a_leaf", ONE_B, "--a_sims", "{SIMS}",
            "--c_puct", str(c), "--tau", str(tau),
            "--b", W1B, "--b_leaf", ONE_B, "--b_sims", "{SIMS}",
            "--b_c_puct", str(BOT[0]), "--b_tau", str(BOT[1]),
            "--positions", "{POSITIONS}", "--slice", f"{{card}}/2",
            "--bag", DATA + "/test.bag", "--device", "cuda", "--threads", "2",
            "--seed", "0", "--hours", str(hours), "--out", out]
    cards.append(launch(args, card, out[:-5]))
codes = [finish(*p) for p in cards]
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 5 <= end_h <= 11.5:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h; tuning takes "
                         "~1.5h and the match needs the rest")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "mcts_tune64_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), tune_cell(), match_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "mcts_tune64_kernel.ipynb", "dataset_sources": [],
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
