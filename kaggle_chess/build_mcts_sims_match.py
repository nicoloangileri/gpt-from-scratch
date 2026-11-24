"""One GPU session: the bot's tree search at 64 simulations against itself at 32.

    python kaggle_chess/build_mcts_sims_match.py --quota_h 9
    kaggle kernels push -p kaggle_chess_mcts_sims_match/

results/39 put the tree search with 32 simulations 115 Elo (101 to 130) above
alpha-beta at depth 2, and the bot now plays it. On the Kaggle CPU the bot plays on, 32
simulations take 2.5 s per move (results/40), so 64 would take about 5 s, which still
fits a 10+3 clock. Is it worth it?

Both sides are the bot: the 1B model's w0.3 fine-tune at the root, the 1B model below,
c 0.3 and prior temperature 0.05 (the c results/39 chose on puzzles; not tuned again).
The 800 positions of results/33-39 (seed 0), each played twice with colours swapped,
split over both cards by `--slice`. A game costs about 2.3 times one of results/39, so
the session's deadline cuts the list: each card stops cleanly between positions and
keeps an even sample of it. Decided before the games: the bot moves to 64 simulations
only if the 95% interval, clustering the two games of each position, excludes zero in
64's favour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_long_run import ROOT, _notebook  # noqa: E402
from build_mcts_match import (ONE_B_KERNEL, POSITIONS, TAU, W1B_KERNEL,  # noqa: E402
                              setup_cell, start_cell)

DIR = "kaggle_chess_mcts_sims_match"
KERNEL_ID = "nicolangileri/gpt-from-scratch-chess-mcts-sims-match"
SIMS_A, SIMS_B = 64, 32
C_PUCT = 0.3
END_BEFORE_QUOTA_H = 0.08
MATCH_EARLY_H = 0.10


def match_cell() -> str:
    return f'''\
hours = hours_left() - {MATCH_EARLY_H}
print(f"[session] match from {{(time.time() - T0) / 3600:.2f}}h, {{hours:.2f}}h per card",
      flush=True)
cards = []
for card in (0, 1):
    out = f"match_mcts{SIMS_A}_vs_mcts{SIMS_B}_part{{card}}.json"
    args = ["src.match", "--a", W1B, "--a_leaf", ONE_B, "--a_sims", "{SIMS_A}",
            "--b", W1B, "--b_leaf", ONE_B, "--b_sims", "{SIMS_B}",
            "--c_puct", "{C_PUCT}", "--tau", "{TAU}",
            "--positions", "{POSITIONS}", "--slice", f"{{card}}/2",
            "--bag", DATA + "/test.bag", "--device", "cuda", "--threads", "2",
            "--seed", "0", "--hours", str(hours), "--out", out]
    cards.append(launch(args, card, out.split(".")[0]))
codes = [finish(*p) for p in cards]
print(f"[session] done at {{(time.time() - T0) / 3600:.2f}}h: {{codes}}", flush=True)
'''


def build(quota_h: float, out_dir: Path) -> Path:
    end_h = round(quota_h - END_BEFORE_QUOTA_H, 2)
    if not 4 <= end_h <= 11.5:
        raise SystemExit(f"--quota_h {quota_h}: jobs would end at {end_h}h")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "mcts_sims_match_kernel.ipynb").write_text(json.dumps(_notebook(
        [setup_cell(), start_cell(end_h), match_cell()]), indent=1))
    meta = json.loads((ROOT / "kaggle_chess" / "kernel-metadata.json").read_text())
    meta.update({"id": KERNEL_ID, "title": KERNEL_ID.split("/")[-1],
                 "code_file": "mcts_sims_match_kernel.ipynb", "dataset_sources": [],
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
