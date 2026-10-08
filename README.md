# gpt-from-scratch

A decoder-only Transformer **built from scratch in PyTorch**, pointed at chess, then
made **fast** -- with reproducible configs, benchmarks, and the run logs behind the
numbers in [`results/`](results/).

Every block (attention, MLP, LayerNorm, positional embeddings, the AdamW param
grouping) is hand-written in [`src/model.py`](src/model.py) -- no `from transformers
import` of the model. The fused LayerNorm in [`src/kernels/`](src/kernels/) is
hand-written too, in Triton.

Hand-written is not the same as unprecedented, so: the model's skeleton follows
Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT) and the fused kernel's
two-stage backward follows Triton's `05-layer-norm` tutorial (MIT). The chess task,
the action-value head, the data path, the search and the bot are this project's.
Ideas reimplemented from DeepMind's ChessBench are listed in [`NOTICE`](NOTICE).

## The result

The model reads a chess position and a candidate move and predicts how good the move
is, learning from Stockfish 16 annotations (DeepMind's ChessBench). No search, no
rollouts -- one forward pass per legal move, best expected value wins. 15M parameters
(14.19M non-embedding), 8 layers, 384 dim, a 1,999-symbol vocabulary (31 FEN symbols
plus 1,968 UCI moves), 78 tokens per sample (77 for the position, 1 for the move), and
a 64-way classification head over buckets of the Stockfish win probability.

**64.8% of Lichess puzzles solved** -- 6,484 of the 10,000 ChessBench test puzzles --
after 500M training examples of Stockfish annotations on free Kaggle GPUs
([`results/24_chess_500M_eval.json`](results/24_chess_500M_eval.json)); 60.3% at 225M.

**77.5%** (7,752 of 10,000) after a 4-hour fine-tune of the 500M checkpoint on *other*
Lichess puzzles ([`results/26_puzzle_finetune_500M_w0.3.json`](results/26_puzzle_finetune_500M_w0.3.json)),
with a ranking loss over the legal moves added on top of the original one at 0.3 of its
weight; at full weight the same fine-tune took the 225M checkpoint to 74.6%. The 10,000
test puzzles were excluded by id before packing (9,995 of them are in the Lichess
database; the packer asserts that the exclusion removed something), and the checkpoint
was chosen by the action-value validation loss, never by puzzle accuracy. All of these
numbers come from one evaluator, which reproduced earlier measurements exactly when it
was rerun (results/17).

Those numbers answer different questions. 64.8% is what Stockfish's annotations alone
teach, which is the setting of DeepMind's paper. 77.5% is what the model does once it
has also been told which move solves a puzzle -- more useful, and no longer comparable
to the paper, which never trains on puzzles.

Neither is readable without a baseline, which is why the headline here is puzzle
accuracy and not a validation loss. Two baselines run on the same 10,000 puzzles: a
ten-line rule that grabs the most valuable hanging piece, or mates when it can, solves
**27.4%**, and picking a legal move at random solves **0.6%**.

### By difficulty

| puzzle rating | puzzles | grab-the-piece | 100M examples | 225M examples | 500M examples | **500M + light puzzle fine-tune** |
|---|---|---|---|---|---|---|
| 0-499 | 74 | 74.3% | 93.2% | 100.0% | 100.0% | **100.0%** |
| 500-999 | 2,161 | 58.3% | 84.4% | 93.6% | 95.1% | **99.2%** |
| 1000-1499 | 3,367 | 32.0% | 61.6% | 76.3% | 80.1% | **93.9%** |
| 1500-1999 | 2,543 | 12.0% | 30.3% | 42.3% | 51.0% | **72.1%** |
| 2000-2499 | 1,571 | 2.8% | 10.8% | 17.4% | 21.7% | **33.4%** |
| 2500-2999 | 284 | 1.1% | 3.2% | 6.0% | 7.0% | **5.3%** |
| **all** | **10,000** | **27.4%** | **49.1%** | **60.3%** | **64.8%** | **77.5%** |

(A random legal move solves 0.6% overall; it is not broken out by band.)

The fine-tune gained most in the middle, 1000-2500, and nothing at the top: 7.0% to
5.3% on 284 puzzles (6.0% to 4.9% for the 225M fine-tune) is inside the noise of a band
that small. Length is the obvious
suspect: the solver has 1.4 moves to find on average below 500 and 3.6 above 2500, and
every one of them has to be right. That is a correlation, not a measured cause.

The gap widens with difficulty -- at 2000-2499 the 500M model solves 21.7% against the
heuristic's 2.8%, nearly eight times as many -- and that is the part which cannot be
explained by capturing hanging pieces. Note also what the baseline exposes at the other
end: 20.1% of the puzzles need a single move from the solver, usually a capture, which
is exactly the move the ten-line rule finds.

### The curve

The point of the project is the curve, not the single number -- how much accuracy an
hour of free GPU buys:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/curve_dark.svg">
  <img alt="Puzzle accuracy against training examples seen: flat near 20% up to 20M examples and below the 27.4% heuristic, then 30.3% at 50M, 49.1% at 100M, 60.3% at 225M and 64.8% at 500M; puzzle fine-tunes take the 225M model to 74.6% and the 500M model to 77.5%" src="results/curve_light.svg">
</picture>

| model | examples | T4 hours | puzzles solved | val loss |
|---|---|---|---|---|
| 15M | 1M | 0.2 | 19.9% | 3.234 |
| 15M | 6.1M | 0.9 | 20.5% | 3.088 |
| 15M | 10M | 1.5 | 20.9% | 3.036 |
| 15M | 20.5M | 3.9 | 21.4% | 2.911 |
| 15M | 30.7M | 5.3 | 23.7% | 2.751 |
| 15M | 50M | 10.0 | 30.3% | 2.5432 |
| 15M | 100M | 21.4 | 49.1% | 2.2419 |
| 15M | 225M | -- | 60.3% | 2.0557 |
| 15M | **500M** | 23.8 on two T4s | **64.8%** | **1.9371** |
| 15M | 225M + 12k puzzle steps | 4.2 | 74.6% | 2.0562 |
| 15M | 500M + 12k puzzle steps, ranking at 0.3 | 4.1 | **77.5%** | 1.9395 |
| 60M | 14.6M | 9.5 (whole run) | 21.4% | 2.986 |

Every row is measured on all 10,000 puzzles, in one Kaggle session with one evaluator
([`results/17_eval_on_kaggle.json`](results/17_eval_on_kaggle.json)); the four rows
that had been measured before came out identical. Earlier drafts had the small runs on
the first 2,000 puzzles only, which turned out half a point to a point easier. The
T4 hours are wall clock from the run logs; the 225M run is warm-started from the 100M
checkpoint and split across two Kaggle sessions, so a single wall-clock figure would
not be comparable -- it held 1,830 samples/s at 20.3% MFU throughout. The 500M run is
warm-started from the 225M one and ran on both T4s of a Kaggle session (see Systems):
three sessions of 11.3, 4.0 and 8.6 hours, the last including the final evaluation, at
3,240-3,450 samples/s by session median ([`results/24_chess_500M_session1.log`](results/24_chess_500M_session1.log)
and the next two). On one T4 its training alone would have taken about 42 hours. The
fine-tune rows are training time only: each session first spent 82-84 minutes fetching
and packing its data, most of it the 6.1M-puzzle database. Their validation losses are
on the same split as the others but from half as many batches (25, not 50), so read
them as "unchanged", not as a comparison to four decimals.

**Every model below 50M examples scores worse than the ten-line rule.** That is the
single most useful thing the baseline did.

Returns are diminishing but not gone: doubling from 50M to 100M examples bought 18.8
points, 2.25x more on top of that a further 11.2, and 2.2x more again 4.5. The
validation loss slowed down less: it fell by 0.119 from 225M to 500M examples, against
0.186 the step before. The model is still learning the annotations; the puzzles are
rewarding it less for each unit of loss. Data is not what is running out -- the 225M
run used 82,195,730 positions from 10 of ChessBench's 2,148 shards and went over them
about 1.5 times; the 500M run added 285M positions from shards 10-41, never used
before, drawn with replacement at about one draw per position (so roughly 4 in 10 were
never drawn at all).

### Per move, not per puzzle

Puzzle accuracy is pass/fail on a whole line. Move by move, on every fully annotated
position of ChessBench's test shard -- 61,833 positions, 1.84M scored moves
([`src/agreement_eval.py`](src/agreement_eval.py)):

| model | action accuracy | picks *a* best move | Stockfish top 3 | Kendall's tau | win prob. given up | moves losing > 15 pts |
|---|---|---|---|---|---|---|
| 1M | 8.5% | 10.9% | 25.3% | 0.190 | 14.2 pts | 29.9% |
| 10M | 14.7% | 17.1% | 34.7% | 0.282 | 11.4 pts | 23.3% |
| 30.7M | 22.2% | 24.7% | 45.9% | 0.381 | 8.4 pts | 16.4% |
| 50M | 30.1% | 32.9% | 55.4% | 0.470 | 5.7 pts | 10.5% |
| 100M | 39.7% | 42.9% | 67.4% | 0.576 | 3.3 pts | 5.2% |
| 225M | 45.2% | 48.6% | 73.7% | 0.632 | 2.3 pts | 3.1% |
| 225M + puzzle fine-tune | 43.1% | 46.5% | 70.4% | 0.619 | 3.0 pts | 5.0% |
| 225M + light puzzle fine-tune | 44.4% | 47.7% | 72.0% | 0.628 | 2.7 pts | 4.2% |
| **500M** | **48.2%** | **51.7%** | **76.8%** | **0.660** | **1.9 pts** | **2.3%** |
| 500M + light puzzle fine-tune | 47.0% | 50.4% | 74.7% | 0.653 | 2.2 pts | 3.3% |

Action accuracy and Kendall's tau (tau-b, over all legal moves, averaged per position)
are the two metrics the ChessBench paper uses in its ablations. Action accuracy is the
paper's definition, the argmax of Stockfish's values; "picks *a* best move" also accepts
a move tied with it, which matters in 5.9% of these positions because win probability
saturates and several moves share the top value. From 50M to 500M examples the rate of
moves that throw away more than 15 points fell from one in ten to one in thirty-two at
225M and one in forty-four at 500M.

**The puzzle fine-tune is not free.** It bought 14.3 points on puzzles and paid for them
in ordinary positions: action accuracy down 2.1 points, and the blunder rate up from
3.1% to 5.0% -- back to the 100M model's level. The action-value loss that ran alongside
the ranking loss limited the damage (tau barely moved, 0.632 to 0.619) but did not
prevent it. With 61,833 positions these differences are far outside the noise.

Turning the ranking loss down removes part of that cost, unevenly. The same fine-tune
with the ranking loss at 0.3 instead of 1.0 -- same data, steps, schedule and card,
only that number changed -- solves **74.4%** of the puzzles (7,436 against 7,455: the
gain is kept). Of the damage it removes about 60% of the action-accuracy drop (0.8
points instead of 2.1) and 70% of tau's (0.003 instead of 0.013), but only 40% of the
extra blunders: 4.2% instead of 5.0%, still 35% more than the 225M model's 3.1%
([`results/20_puzzle_finetune_w0.3.json`](results/20_puzzle_finetune_w0.3.json); the
training logs of both runs are next to it). The blunder differences are 6 to 10
standard errors wide even without pairing the models position by position; the
0.8-point action-accuracy drop is about 3. It is one training run per weight, and a
fine-tune with no ranking loss at all has not been run, so how much of what remains is
the cost of the ranking loss and how much the cost of fine-tuning at all is open.

The same light fine-tune on the 500M model behaves the same way
([`results/26_puzzle_finetune_500M_w0.3.json`](results/26_puzzle_finetune_500M_w0.3.json)):
12.7 points on puzzles (64.8% to 77.5%), for action accuracy down 1.2 and blunders up
from 2.3% to 3.3% -- where the 225M model was before any fine-tune. Its validation loss
took the same detour, up 0.10 in the first thousand steps and back by the end.

And they cost games. In a head-to-head match without search
([`src/match.py`](src/match.py): balanced middlegame starts from the held-out test shard,
each played twice with colours swapped), the model **before** the fine-tune beat the one
after it **174-59 with 167 draws over 400 games: +103 Elo, 95% interval +77 to +130**.
Against the light fine-tune, on the same 200 starting positions, it wins 150-93 with 157
draws: **+50 Elo, interval +23 to +77**
([`results/22_match_400_depth1_light.json`](results/22_match_400_depth1_light.json)).
Turning the ranking loss down to 0.3 halved the cost in play, and paired by starting
position the gap between the two matches is 3.0 standard errors -- roughly in line
with the blunders, whose excess over the 225M model it cut by 40%.
With depth-2 search on both sides, on the same 200 starting positions, the full
fine-tune's gap nearly closes: **134-114 with 152 draws, +17 Elo, 95% interval -9 to +44**
([`results/21_match_400_depth2.json`](results/21_match_400_depth2.json)). The two
intervals do not overlap, and paired by starting position the score drop from depth 1
to depth 2 is 4.7 standard errors. Search absorbs most of the fine-tune's cost in
play -- which fits the move-quality table, since the extra errors are one-move
blunders and those are exactly what a two-ply search looks at, though that part is an
explanation, not a measurement. (Each position's two games are not independent; an
interval computed over the 200 colour-swapped pairs instead of the 400 games comes out
within a point of these.) So "better" depends on the job and on the search: for
finding tactics the fine-tuned model; for playing without search the 225M one, by about
a hundred Elo; with depth-2 search, a difference this match cannot resolve. The
first-to-7 match at depth 2 earlier, 7.5-6.5 to the fine-tuned model over 14 games,
was consistent with that and could not have shown it. The 400 games took 8.0 hours on
a T4; on the laptop's CPU the estimate had been about 30.

More data buys games as well as puzzles. On the same 200 starting positions and without
search, the 500M model beats the 225M one **165-55 with 180 draws: +98 Elo, interval +73
to +124** ([`results/25_match_500M_vs_225M_depth1.json`](results/25_match_500M_vs_225M_depth1.json)),
about the margin by which the 225M model beat its own fine-tuned version.
With depth-2 search on both sides the gap widens instead of closing: **227-45 with 128
draws, +171 Elo, interval +142 to +201**
([`results/27_match_500M_vs_225M_depth2.json`](results/27_match_500M_vs_225M_depth2.json)).
Paired by starting position, the 500M model scores 0.09 a game more at depth 2 than at
depth 1, 3.6 standard errors. Search does the opposite of what it did to the fine-tune's
gap. One reading, not a measurement: what a two-ply search absorbs is the one-move
blunder, which it catches whichever network is evaluating; what it cannot supply is a
better value at the leaves, and that is where the 500M model's gain lies (top-1 agreement
with Stockfish 45.2% to 48.2%, mean absolute error 0.036 to 0.032) -- every leaf of a
depth-2 tree is read through it. The 400 games took 8.0 hours on a T4, like the first
depth-2 match.

**Calibrated from the start; what training buys is sharpness.** When the model predicts
a win probability, Stockfish's own number agrees on average: the binned calibration
error is at most 0.007 for every model, the 1M one included. What improves with data is
how close each prediction lands -- mean absolute error 0.125 at 1M examples, 0.036 at
225M and 0.032 at 500M, 3.9x smaller. The full predicted distribution is slightly conservative: its
central 50% interval contains Stockfish's value 50-55% of the time and its 90% interval
91-92%, a little more as training goes on.

### Capacity was the wrong lever, and the experiment is in the table

Chinchilla scaling puts the optimum for a 9-hour T4 budget well above 15M parameters;
the 60M model scored 21.4% at 14.6M examples, level with the 15M model at 20.5M
examples and below the same model at 30.7M. Scaling laws are not wrong: the larger model is
roughly 4x slower per step, so it sees far fewer examples per hour, and on this budget
compute per hour is the binding constraint, not parameter count. That would flip on
better hardware -- exactly the kind of thing papers running on clusters never have to
report.

### Distance to the paper

DeepMind's ChessBench reaches **88.9%** on these puzzles with a **9M**-parameter model
-- smaller than this one, on the same data and the same task. The difference is
training examples seen: theirs is roughly 600x this budget. 64.8% from free T4s is a
budget result, not an architecture result, and the curve above is the honest way to
read it.

## Search and the Lichess bot

[`src/search.py`](src/search.py) puts alpha-beta (negamax) on top of the network. The
network returns Q(s,a) for every legal move in one batch, which is both the leaf value
(V(s) = max Q) and the move ordering that alpha-beta's pruning lives on: one network
call per node, not one per move. A move's value is the **minimum** of the direct
Q(s,a) and the value backed up from the child, because a maximiser otherwise hunts for
wherever the evaluator is most wrong -- out-of-distribution positions where the network
answers near the mean.

Depth 2 beats the bare network **10-0**, every game by checkmate, over 5 random
openings times 2 colours. It costs 1.5-3 s per move on CPU against 0.03 s for the bare
network.

On puzzles it does the opposite. Same fine-tuned checkpoint, same first 1,000
puzzles, the only difference the search: **74.3% at depth 1, 70.9% at depth 2**
([`results/20_puzzle_finetune_w1.0.log`](results/20_puzzle_finetune_w1.0.log)). The
loss is concentrated where puzzles are easy -- 500-999 drops from 99.0% to 93.7%,
1000-1499 from 93.2% to 86.6% -- while 1500-2499 is level. A game rewards not
hanging pieces over a two-move horizon; a puzzle rewards trusting one move, and a
backed-up value from a position the network sees less clearly can overrule a root
judgement that was right.

Before the fine-tune it was the other way round: on the 225M checkpoint, same 1,000
puzzles, **58.5% at depth 1 and 60.8% at depth 2** -- search helped. So the fine-tune
is what flipped it, and the move-quality table says how: the fine-tune made the values
of ordinary positions worse (blunders 3.1% to 5.0%), and depth-2 search is built out
of exactly those values, read in the position after each candidate move. On their own,
differences of 2-3 points on 1,000 puzzles are suggestive rather than conclusive; what
makes the explanation credible is that it agrees with a measurement on 61,833 positions
that was taken for a different reason. (An earlier depth-2 figure was measured with the
puzzle-parity bug described below and was withdrawn.)

The light fine-tune, with the ranking loss at 0.3, sits in between on both counts: 4.2%
blunders, and on the same 1,000 puzzles **73.2% at depth 1 and 72.6% at depth 2**. In
order -- no fine-tune, ranking loss at 0.3, ranking loss at 1.0 -- search changes the
puzzle score by +2.3, -0.6 and -3.4 points while the blunder rate goes 3.1%, 4.2%,
5.0%. Each of those changes carries a standard error of one to two points on 1,000
puzzles, so this is a trend, not a proof, but it runs the way the explanation
predicts.

A fourth point does not fit as neatly. The light fine-tune of the 500M model blunders
less (3.3%) and still loses 2.1 points to search on the same 1,000 puzzles (77.0% to
74.9%), more than the 225M one did at 4.2%. That is inside the same noise, and the 500M
model's own depth-1-against-depth-2 figure before any fine-tune -- the baseline this
point would need -- has not been measured; until it is, the explanation above stays a
hypothesis that three points of four agree with.

[`src/lichess_bot.py`](src/lichess_bot.py) plays the same search online, at depth 2, as
**`imbotterthanyou`**: rapid **1659** on 8 October 2026. Every game in the bot's log,
matched to the checkpoint that played it
([`results/28_lichess_bot_by_model.json`](results/28_lichess_bot_by_model.json)): the
100M model took the rating from 1270 to 1461 in 10 games, the 225M one to 1634 in 55, the
fully fine-tuned 225M played 6 (3 won, 3 lost), the 225M 12 more to 1644, and the 500M
one 24 -- 11 won, 3 drawn, 10 lost -- to 1659. As a performance rating against the
opponents' pre-game ratings, the 500M model's 1701 beats the 225M one's 1678 by 23, with
a bootstrap interval of -150 to +187. A few hours online agree with the depth-2 match
and could not have measured it: the opponents change from game to game, and 24 games
leave an interval 300 Elo wide. None of this is comparable to the paper's 2054: that is blitz against bots, this is rapid
against bots -- a different pool at a different time control. Search depth
adapts to the clock -- the bot declines time controls it cannot play at depth 2 and
drops to the bare network when short on time. The API token comes from `LICHESS_TOKEN`
or from `~/.config/lichess/token` with owner-only permissions, which the bot checks and
refuses if they are wider; never a command-line argument, never logged, never inside
the repository.

Running a bot is mostly a lesson in failure modes that look like success. A thread that
died on one dropped connection left the process up, the log open, and the bot silently
challenging nobody for four hours. Catching a network error during a game and returning
kept the process alive and *forfeited* two rated games, since Lichess keeps the clock
running: the fix is to reconnect, not to survive. The same clock makes stopping the bot
a decision about a game: a stop request (`pkill` or Ctrl+C) with a game on the board
declines new challenges and leaves when that game is over; only a second request leaves
at once. And a bot cannot enter the lobby or a
tournament -- the API spec says pools and tournaments are off-limits -- so when the bot
pool has spent its 100 daily games, the only lever left is to offer faster time controls
and remember who already said no.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # CPU, ~25s: 88 passed, 11 skipped (Triton needs CUDA)
```

Chess, one shard end to end (1.34 GB download, packs to .bin, trains, scores puzzles):

```bash
mkdir -p data/chessbench
BASE=https://storage.googleapis.com/searchless_chess/data
curl -o data/chessbench/action_value-00000.bag \
     $BASE/train/action_value-00000-of-02148_data.bag
curl -o data/chessbench/puzzles.csv $BASE/puzzles.csv

python -c "from src.chess_data import prepare_chessbench; \
           prepare_chessbench(['data/chessbench/action_value-00000.bag'], 'data/chess')"
python -m src.train_chess --config configs/chess.yaml
python -m src.puzzles_eval --ckpt checkpoints/chess_best.pt \
       --puzzles data/chessbench/puzzles.csv --num 10000
```

Any key in the YAML can be overridden on the command line (`--device cpu`,
`--max_steps 5000`, `--batch_size 1024`); an unknown key is reported, not silently
ignored. Training writes `checkpoints/chess_best.pt`.

The language-model path the project started from still works:

```bash
python -c "from src.data import prepare; prepare('data/tinystories.txt', 'data/tinystories')"
python -m src.train --config configs/config.yaml
python -m src.sample --ckpt checkpoints/best.pt --prompt "Once upon a time"
```

Runs are one config plus one command, seeds fixed. GPU runs are assembled into Kaggle
notebooks by [`kaggle_chess/build_chess_notebook.py`](kaggle_chess/build_chess_notebook.py)
and, for runs longer than one 12-hour session,
[`kaggle_chess/build_long_run.py`](kaggle_chess/build_long_run.py).

## Systems

Throughput on a T4, measured with [`src/bench_chess.py`](src/bench_chess.py)
(each configuration in its own subprocess):

| change | samples/s | MFU |
|---|---|---|
| baseline | 1537 | 17.1% |
| + fused AdamW | 1750 | 19.5% |
| + torch.compile, batch 1024 | **1849** | **20.6%** |

The fused LayerNorm ([`src/kernels/layernorm.py`](src/kernels/layernorm.py)) against
PyTorch, at the model's shapes:

| rows | eager | torch.compile | Triton | bandwidth |
|---|---|---|---|---|
| 9,984 | 1.00x | 0.93x | 0.86x | 147 GB/s |
| 39,936 | 1.00x | 1.71x | **1.86x** | 298 GB/s |
| 79,872 | 1.00x | 1.40x | **1.48x** | 295 GB/s |

295-298 GB/s is about 92% of the T4's 320 GB/s ceiling, so for a memory-bound op there
is little left on the table: the kernel is 1.5-1.9x over eager and 5-8% over
torch.compile at the sizes that matter. Below ~10k rows it *loses* to eager -- launch
overhead outweighs the win -- which is why the PyTorch fallback stays in.

The honest ceiling: end to end that kernel is worth about **3%** (a 2,000-step A/B on
the T4 ran 681 s with the PyTorch LayerNorm and 661 s with the Triton one,
[`results/07_triton_layernorm.log`](results/07_triton_layernorm.log)). LayerNorm is a
small share of the step. Amdahl. And the GEMMs that dominate it are not reachable from
here either: on a T4 torch.compile logs `Not enough SMs to use max_autotune_gemm mode`
and skips the autotuned GEMM templates altogether, so what `compile` buys on this card
is fusion and fewer launches.

### Two GPUs, not one

Kaggle's "GPU" accelerator is two T4s -- a single T4 is not offered (kaggle-cli PR
1198) -- and the weekly quota counts the session, not the cards. Every run above asked
for that accelerator (`NvidiaTeslaT4`) and trained on one card: the probe below saw
both, the earlier logs never counted them. The same session, the same data, the same starting
checkpoint, measured back to back
([`results/23_one_vs_two_t4.log`](results/23_one_vs_two_t4.log)):

| | samples/s | MFU per GPU |
|---|---|---|
| one T4 | 1,907 | 21.2% |
| two T4s, DDP, global batch still 1024 | **3,546** | 19.8% |

1.86x, 93% scaling efficiency, at no extra quota (every log line from step 200 on, once
compilation is over). The ranks read disjoint slices of the data and seed the same RNG,
so the RNG state one rank saves is every rank's and a run split across sessions still
matches an unsplit one bit for bit (`tests/test_ddp.py`, on two gloo ranks).

The same numbers settle what an earlier draft of this section got wrong. It guessed
that the step was bound by kernel launches and wanted CUDA graphs next. A launch-bound
step costs the same whatever the batch; here each card's step fell from 0.537 s to
0.289 s when its batch halved from 1,024 to 512. The fixed part -- launches and the
gradient all-reduce together -- is at most 0.289 - 0.537/2 = 20 ms of 289, about 7%, and
that is all CUDA graphs could win. The step is bound by the work itself.

## What was learned the hard way

Documented because the mistakes were more instructive than the successes:

- **A warm restart is not free.** The 225M run started from the 100M checkpoint with a
  fresh cosine at lr 5e-4. Validation loss went *up*, from 2.2412 to 2.4834 in 8,000
  steps, and it took about 52,000 steps just to get back to where it started. All of
  the gain arrived in the final third, as the learning rate decayed. The lesson is
  budgetary: if the budget cannot cover the whole decay, restarting is worse than not
  restarting. It held at a fifth of the learning rate too: the puzzle fine-tune, at
  1e-4, still lifted the validation loss from 2.043 to 2.150 in its first 3,000 steps
  and only got it back as the schedule decayed. The 500M run restarted at 3e-4 instead
  of 5e-4 and its detour was smaller, 2.051 to 2.247, with one spike to 2.781 just after
  the warmup that took 8,000 steps to settle; it was back below where it started at 64%
  of the schedule and ended at 1.937.
- A throughput number that counts only one of two batches halves itself. The fine-tune
  logged 822 samples/s at 9% MFU because the puzzle batch -- as many sequences again --
  was not counted; it looked exactly like a regression.
- `torch.cuda.is_bf16_supported()` counts emulation and answers `True` on a T4, where
  bf16 is emulated and slow. Detect the card by compute capability instead: worth 4.1x.
- The same metric was got wrong twice, in opposite directions. Lichess puzzle files
  give the position *before* the opponent's move, so the solver plays the odd plies.
  A first version read it right; a "fix" then flipped it, on the evidence that
  `Moves[0]` matched the first move of the `Solution` column -- which is that same
  line written in another notation, so the check could only agree. For ten days every
  reported accuracy was the model guessing the *opponent's* moves. What settled it:
  ChessBench's file is byte-identical to `lichess_db_puzzle` for the same PuzzleId,
  and DeepMind's own evaluator asks the engine on odd plies. The lesson is not about
  chess: **a test that compares a field with a restatement of itself will certify
  anything.** The replacement asks who ends up checkmated.
- A metric without a baseline is not a measurement. Until a ten-line heuristic was run
  on the same puzzles, there was no way to know that most models here were below it.
- Timing a training step without `torch.cuda.synchronize()` measures how long the CPU
  took to *queue* the work. It reported an MFU of 882%.
- Benchmarking configurations in one process lets the first one leave the allocator
  fragmented; a larger batch then looks slower, and OOMs on an allocation the card
  has room for.
- A CPU-only test cannot catch a device-placement bug: resuming loaded the whole
  checkpoint onto the GPU, RNG state included, and `torch.set_rng_state` refused it.
  Running the same test on MPS reproduced it locally.
- A mutation check can be fooled by a stale `.pyc`: the mutated and restored files had
  the same size and the same mtime second, so Python reused the old bytecode and the
  "restored" run still failed.
- Read the hardware before tuning for it. For a month the runs here asked for Kaggle's
  GPU accelerator, which is two T4s, and trained on one, while kernels and batch size
  were tuned for "the T4". No log counted the cards; `torch.cuda.device_count()` would
  have said 2 on the first day.
- Deeper search does not average out evaluator error, it amplifies it. Alpha-beta
  actively searches for the positions where the network is most wrong, which is why a
  move is scored by the minimum of its two estimates rather than the backed-up one.

## Layout

```
src/model.py         hand-written GPT (attention, MLP, LayerNorm, generate)
src/chess_data.py    ChessBench .bag reader, FEN tokenizer, 1968-move vocabulary
src/train_chess.py   action-value training loop (resume, warm restart, deadlines)
src/puzzles_eval.py  Lichess puzzle accuracy
src/puzzle_data.py   Lichess puzzle database -> a ranking task (held-out ids excluded)
src/agreement_eval.py  per-move agreement with Stockfish
src/search.py        alpha-beta on top of the network
src/match.py         head-to-head matches between two checkpoints, Elo with an interval
src/lichess_bot.py   plays online; token from the environment or an owner-only file
src/kernels/         fused LayerNorm in Triton + its bench
src/bench_chess.py   throughput bench (isolated subprocess per config)
src/data.py          BPE tokenizer + streaming corpus packing
src/train.py         language-model training loop
kaggle_chess/        builds the Kaggle notebooks (training, fine-tunes, evaluation, matches)
results/             the logs behind the numbers above
```

## Still open

- Whether the 500M model's light fine-tune -- with the 225M model's blunder rate -- also
  plays at the 225M model's level (a match would say)
- Search on the 500M model's puzzles before any fine-tune (depth 1 against depth 2), the
  baseline the fourth search point above is missing
- A fine-tune with the ranking loss switched off, to split what remains of the cost
  between the ranking loss and the fine-tune itself
- KV-cache serving for the language-model path (CUDA graphs: worth at most ~7% here,
  see Systems)
