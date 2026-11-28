# Experiments, in full

The [README](README.md) keeps the main path. This file keeps every experiment behind it,
with its caveats, in the order they were run; each number links to its log in
[`results/`](results/).

- [Puzzles and training](#puzzles-and-training)
- [Search and the Lichess bot](#search-and-the-lichess-bot)
- [Systems](#systems)
- [What was learned the hard way](#what-was-learned-the-hard-way)

## Puzzles and training

### By difficulty

| puzzle rating | puzzles | grab-the-piece | 100M examples | 225M examples | 500M examples | 1B examples | 500M + light puzzle fine-tune | **1B + light puzzle fine-tune** |
|---|---|---|---|---|---|---|---|---|
| 0-499 | 74 | 74.3% | 93.2% | 100.0% | 100.0% | 98.6% | 100.0% | **100.0%** |
| 500-999 | 2,161 | 58.3% | 84.4% | 93.6% | 95.1% | 97.1% | 99.2% | **99.4%** |
| 1000-1499 | 3,367 | 32.0% | 61.6% | 76.3% | 80.1% | 83.3% | 93.9% | **94.9%** |
| 1500-1999 | 2,543 | 12.0% | 30.3% | 42.3% | 51.0% | 55.9% | 72.1% | **73.8%** |
| 2000-2499 | 1,571 | 2.8% | 10.8% | 17.4% | 21.7% | 24.6% | 33.4% | **36.3%** |
| 2500-2999 | 284 | 1.1% | 3.2% | 6.0% | 7.0% | 6.3% | 5.3% | **8.5%** |
| **all** | **10,000** | **27.4%** | **49.1%** | **60.3%** | **64.8%** | **68.0%** | **77.5%** | **78.9%** |

(A random legal move solves 0.6% overall; it is not broken out by band.)

The fine-tune gained most in the middle, 1000-2500, and nothing at the top: 7.0% to
5.3% on 284 puzzles (6.0% to 4.9% for the 225M fine-tune, 6.3% to 8.5% for the 1B one)
is inside the noise of a band
that small. Length is the obvious
suspect: the solver has 1.4 moves to find on average below 500 and 3.6 above 2500, and
every one of them has to be right. That is a correlation, not a measured cause.

The gap widens with difficulty -- at 2000-2499 the 500M model solves 21.7% against the
heuristic's 2.8%, nearly eight times as many -- and that is the part which cannot be
explained by capturing hanging pieces. Note also what the baseline exposes at the other
end: 20.1% of the puzzles need a single move from the solver, usually a capture, which
is exactly the move the ten-line rule finds.

### Notes on the curve

The table is in the README ([The curve](README.md#the-curve)).

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
and the next two). On one T4 its training alone would have taken about 42 hours. The 1B
run is warm-started from the 500M one with a fresh schedule: five sessions of 11.3, 11.3,
7.1, 11.3 and 1.5 hours on both T4s, the last including the final evaluation, at
3,160-3,660 samples/s by session median
([`results/36_chess_1B_session1.log`](results/36_chess_1B_session1.log) and the next four). The
fine-tune rows are training time only: each session first spent 82-88 minutes fetching
and packing its data, most of it the 6.1M-puzzle database. Their validation losses are
on the same split as the others but from half as many batches (25, not 50), so read
them as "unchanged", not as a comparison to four decimals.

**Every model below 50M examples scores worse than the ten-line rule.** That is the
single most useful thing the baseline did.

Returns are diminishing but not gone: doubling from 50M to 100M examples bought 18.8
points, 2.25x more on top of that a further 11.2, 2.2x more again 4.5, and 2x more
3.2. Up to 500M the validation loss slowed down less than the puzzles did: it fell by
0.186 from 100M to 225M examples and 0.119 from 225M to 500M, so each unit of loss bought
fewer puzzles. The last doubling reversed that: the loss fell by only 0.036, for 3.2
points. Data is not what is running out -- the 225M
run used 82,195,730 positions from 10 of ChessBench's 2,148 shards and went over them
about 1.5 times; the 500M run added 285M positions from shards 10-41, never used
before, drawn with replacement at about one draw per position (so roughly 4 in 10 were
never drawn at all), and the 1B run drew 500M more from the 505M positions of shards
42-121, again never used before.

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
| 500M | 48.2% | 51.7% | 76.8% | 0.660 | 1.9 pts | 2.3% |
| 500M + light puzzle fine-tune | 47.0% | 50.4% | 74.7% | 0.653 | 2.2 pts | 3.3% |
| 500M + the same steps, no puzzles | 48.5% | 51.9% | 76.9% | 0.662 | 1.9 pts | 2.2% |
| 500M + light fine-tune, distilled | 47.7% | 51.1% | 75.8% | 0.658 | 2.1 pts | 2.8% |
| **1B** | **49.5%** | **53.0%** | **77.9%** | **0.671** | **1.8 pts** | **2.1%** |
| 1B + light puzzle fine-tune | 48.4% | 51.8% | 75.9% | 0.665 | 2.1 pts | 2.8% |

Action accuracy and Kendall's tau (tau-b, over all legal moves, averaged per position)
are the two metrics the ChessBench paper uses in its ablations. Action accuracy is the
paper's definition, the argmax of Stockfish's values; "picks *a* best move" also accepts
a move tied with it, which matters in 5.9% of these positions because win probability
saturates and several moves share the top value. From 50M to 500M examples the rate of
moves that throw away more than 15 points fell from one in ten to one in thirty-two at
225M, one in forty-four at 500M and one in forty-eight at 1B.

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
fine-tune with no ranking loss at all was not run on the 225M model; on the 500M one it
was, below, and it says the cost is the ranking loss's.

The same light fine-tune on the 500M model behaves the same way
([`results/26_puzzle_finetune_500M_w0.3.json`](results/26_puzzle_finetune_500M_w0.3.json)):
12.7 points on puzzles (64.8% to 77.5%), for action accuracy down 1.2 and blunders up
from 2.3% to 3.3% -- where the 225M model was before any fine-tune. Its validation loss
took the same detour, up 0.10 in the first thousand steps and back by the end. On the 1B
model too ([`results/37_results_1B_w03.json`](results/37_results_1B_w03.json)): 10.9
points on puzzles (68.0% to 78.9%), for action accuracy down 1.2 and blunders up from
2.1% to 2.8%. The checkpoint kept is step 10,000 of 12,000, chosen as always by the
validation loss, never by puzzles.

The control is the same fine-tune with the puzzles taken out: same start, same three
shards of ChessBench positions, same 12,000 steps, batch, learning rate and schedule, only
no puzzle batch and so no ranking loss
([`results/32_finetune_500M_w0.json`](results/32_finetune_500M_w0.json)). It solves
**64.9%** of the puzzles, against 64.8% before: +0.05 points, standard error 0.28,
paired, 407 puzzles gained and 402 lost. Move by move it is the 500M model again or a
hair better (last row of the table). So all 12.7 points of the light fine-tune, and all
of its cost in ordinary positions, belong to the ranking loss; the extra steps on fresh
positions bought nothing measurable.

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

The 500M model's light puzzle fine-tune, at depth 2 against the 225M model, wins
**140-65 with 105 draws over 310 games: +86 Elo, interval +54 to +118**
([`results/29_match_500M_w0.3_vs_225M_depth2.json`](results/29_match_500M_w0.3_vs_225M_depth2.json)).
The session's deadline stopped it after 155 of the 200 positions. On those same 155 the
500M model before its fine-tune scored 0.707 against the 225M one (+153), so paired by
position the fine-tune costs 0.086 a game, 3.1 standard errors: about 67 Elo, read
through a common opponent rather than played head to head. Search does not erase this
cost the way it nearly erased the 225M model's full fine-tune (+17, interval -9 to +44),
though the two are measured differently, and the one-move-blunder reading above would
have expected it to. The fine-tuned model still plays well above the
225M one, and the bot stays on the 500M model without the fine-tune.

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
is what flipped it. The first explanation came from the move-quality table: the
fine-tune made the values of ordinary positions worse (blunders 3.1% to 5.0%), and
depth-2 search is built out of exactly those values, read in the position after each
candidate move. (An earlier depth-2 figure was measured with the puzzle-parity bug
described below and was withdrawn.)

The light fine-tune, with the ranking loss at 0.3, sits in between on both counts: 4.2%
blunders, and on the same 1,000 puzzles **73.2% at depth 1 and 72.6% at depth 2**. In
order -- no fine-tune, ranking loss at 0.3, ranking loss at 1.0 -- search changes the
puzzle score by +2.3, -0.6 and -3.4 points while the blunder rate goes 3.1%, 4.2%,
5.0%. Each of those changes carries a standard error of one to two points on 1,000
puzzles, so this was a trend, not a proof, and it ran the way the explanation predicted.

The 500M model's figures come from a larger, paired sample: one session put every puzzle
to the 500M model and its light fine-tune, each at depth 1 and depth 2, before moving to
the next (`compare_search` in [`src/puzzles_eval.py`](src/puzzles_eval.py),
[`results/30_search_on_puzzles_500M.json`](results/30_search_on_puzzles_500M.json)),
7,616 puzzles in all. The fine-tune's figures on the first 1,000, 77.0% and 74.9%, came
out identical to the earlier session's. Before the fine-tune, search helps: **62.1% to
64.9% on the first 1,000, and +2.5 points over all 7,616 (standard error 0.24)** -- 271
puzzles solved only at depth 2, 81 only at depth 1. After the light fine-tune it hurts:
**-1.1 points (standard error 0.44)**, 520 against 601. By difficulty the losses sit
where they sat for the 225M full fine-tune, on easy puzzles: search costs the fine-tuned
model 2.0 points at 500-999 and 4.2 at 1000-1499 and gives it 4.7 at 2000-2499, while
the model before the fine-tune loses in no band.

That breaks the blunder-rate explanation. The 500M fine-tune blunders 3.3% of the time,
about as rarely as the 225M model before any fine-tune (3.1%), yet search costs one and
helps the other: what sets the sign is whether the model was fine-tuned on puzzles, not
how often it blunders. The reading that fitted all five points: the ranking loss
teaches the model to put the solving move first at the root, while the values of the
positions after each move -- what depth-2 search reads -- get no such training, so
search lets values that do not see the tactic overrule a root that does. Easy puzzles,
where the root alone is enough, lose the most.

**The test** ([`src/root_ranking.py`](src/root_ranking.py),
[`results/32_root_ranking_500M_w03.json`](results/32_root_ranking_500M_w03.json)). At
depth 2 a move is worth min(Q at the root, value backed up from the position after it),
and nothing forces the two to come from the same network. Depth-1 children never prune,
so the depth-2 choice can be read off one evaluation of every root move and every child
for each network, and every pairing scored at once. That shortcut is checked against
the search itself: the four same-network configurations reproduce results/30 on all
7,616 of its puzzles, puzzle by puzzle, and the w0 checkpoint's depth 1 matches its own
`puzzles_eval` run on all 10,000. On the 10,000 puzzles, paired:

| root | leaves | solved | against the root alone |
|---|---|---|---|
| 500M | none (depth 1) | 64.8% | |
| 500M | 500M | 67.2% | +2.3 (0.22) |
| 500M | fine-tune | 66.7% | +1.8; -0.5 (0.24) against its own leaves |
| fine-tune | none (depth 1) | 77.5% | |
| fine-tune | fine-tune | 76.3% | -1.2 (0.39) |
| fine-tune | 500M | **78.2%** | +0.7 (0.35); +1.9 (0.32) against its own leaves |

The base model's leaves under the fine-tune's root recover 1.9 points, and they were
trained on no more puzzles than the fine-tune's. So the leaves are not merely untrained:
the fine-tune made them worse. The ranks say the same at each puzzle's first solver move.
By Q at the root the fine-tune puts the solution first 86.9% of the time against the base
model's 75.9%; by the backed-up value alone, 81.8% against 84.1%. All of its gain is at
the root, and the value it backs up from the next position got worse -- which is where
the move-by-move table had put the cost, in ordinary positions.

The no-puzzle control settles the remaining question: fine-tuning by itself does not turn
search against the network. Its depth 2 gains 2.5 points (0.21), like the base model's,
its backed-up value ranks the solution first 84.8% of the time, and swapping its leaves
with the base model's changes nothing (-0.05 and +0.12, standard errors 0.16)
([`results/32_root_ranking_500M_w0.json`](results/32_root_ranking_500M_w0.json)).

Two things the reading did not predict. Good leaves help a strong root less: +0.7 under
the fine-tune's root against +2.3 under the base model's, and easy puzzles still lose a
little (99.2% to 98.6% at 500-999, 93.9% to 91.3% at 1000-1499) while 2000-2499 gains
6.4 points. And the blunder rate was the right kind of quantity compared across the wrong
pair: the fine-tune's leaves blunder about as often as the 225M model's, but what decides
the sign is how they compare with the root they can overrule. The hybrid, 78.2%, was the
best puzzle score of the 500M models, two standard errors over the fine-tune alone.

**The hybrid in games** ([`results/33_match_hybrid_vs_500M_d2.json`](results/33_match_hybrid_vs_500M_d2.json)).
The fine-tune's root over the 500M model's leaves against the 500M model alone, both at
depth 2, 400 starting positions played twice with colours swapped: 307 won, 245 drawn,
248 lost, a score of 0.537 and **+26 Elo (95% interval 7 to 45, clustering the two games
of each position)**. It is a small gain, and the two halves of the list disagree on its
size: +42 on the first 200 positions, +10 on the second, with intervals that overlap.
What makes it interesting is the root it uses. Against the 225M model, from the same list
of positions, the fine-tune with its own leaves scored about 85 Elo lower than the 500M model
([`results/29_match_500M_w0.3_vs_225M_depth2.json`](results/29_match_500M_w0.3_vs_225M_depth2.json),
[`results/27_match_500M_vs_225M_depth2.json`](results/27_match_500M_vs_225M_depth2.json));
with the 500M model's leaves it finishes ahead of that same model. The leaves were the
cost in games as well as on puzzles.

**Keeping the leaves** ([`results/33_root_ranking_500M_w03_distill.json`](results/33_root_ranking_500M_w03_distill.json)).
If the ranking loss damages the leaves, a fine-tune that is not allowed to damage them
should get search back. The light fine-tune again, same start, data, steps and schedule,
with one change: every ordinary batch also pays KL(500M || student) over the 64 buckets,
with the 500M model frozen as the teacher. On the same 10,000 puzzles, paired:

| root | leaves | solved | against |
|---|---|---|---|
| distilled | none (depth 1) | 78.1% | +0.56 (0.28) against the fine-tune's depth 1 |
| distilled | distilled | 77.0% | -1.09 (0.36) against its own root |
| distilled | 500M | 78.1% | +0.07 (0.34) against its own root; -0.09 (0.20) against the hybrid |
| 500M | distilled | 67.2% | +0.03 (0.21) against the 500M model at depth 2 |

Distillation did more for the root than for what it was built to protect. The puzzle score
went up rather than down, and blunders fell from 3.3% to 2.8%, about half of the way back
to the base model's 2.3%. Its backed-up value ranks the solution first 82.8% of the time,
between the fine-tune's 81.8% and the base model's 84.1%. But the prediction failed:
search still costs 1.1 points. Even the base model's leaves give this root nothing
measurable (+0.07), where they gave the plain fine-tune's root 0.7. The likely reading is
that a root this strong leaves the leaves little to correct, so keeping them is not enough
to bring search back. At depth 1 it is the best of the 500M networks on puzzles; the
plain fine-tune of the 1B model, 78.9%, is ahead of it.

Does its root beat the fine-tune's in games? Both over the 500M model's leaves at depth 2,
so that only the root differs, 800 positions played twice
([`results/35_match_distilled_vs_hybrid_part0.json`](results/35_match_distilled_vs_hybrid_part0.json)
and `part1`): 593 won, 460 drawn, 547 lost, a score of 0.514 and **+10 Elo for the distilled
root (95% interval -4 to 24)**. The rule was fixed in
[`kaggle_chess/build_match_distill.py`](kaggle_chess/build_match_distill.py) before the
games: the bot changes root only if the interval excludes zero. It does not, so the
fine-tune's root stays. The games agree with the puzzles, where the two were level.

**The hybrid on the 1B model** ([`results/37_match_1B_hybrid_vs_hybrid_part0.json`](results/37_match_1B_hybrid_vs_hybrid_part0.json)
and `part1`). The 1B model's light fine-tune at the root and the 1B model at the leaves,
against the bot's hybrid (the 500M model's fine-tune over the 500M model's leaves), both
at depth 2, the same 800 positions played twice: 696 won, 440 drawn, 464 lost, a score of
0.573 and **+51 Elo (95% interval 37 to 65)**, +55 and +47 on the two halves of the list.
The rule was fixed in [`kaggle_chess/build_1b_puzzle_match.py`](kaggle_chess/build_1b_puzzle_match.py)
before the games, the same as above, and the interval excludes zero: the bot now plays this
pairing. The 1B model alone, at the root and at the leaves, scores **+45 (31 to 59)**
against the same opponent ([`results/37_match_1B_vs_hybrid_d2.json`](results/37_match_1B_vs_hybrid_d2.json)),
so most of the gain comes with the longer run itself. Each side takes a median of 0.06 s
per move on a T4.

**What the root and the search add on the 1B model**
([`results/38_match_1B_hybrid_vs_1B_d2_part0.json`](results/38_match_1B_hybrid_vs_1B_d2_part0.json),
[`results/38_match_1B_hybrid_vs_1B_d1_part0.json`](results/38_match_1B_hybrid_vs_1B_d1_part0.json)
and their `part1`). The bot's pairing against the 1B model alone, both at depth 2, the
same 800 positions played twice: 584 won, 560 drawn, 456 lost, a score of 0.540 and
**+28 Elo (95% interval 14 to 42)**, clustering the two games of each position. The
results above put the two 6 Elo apart, but that is the difference of two estimates each
about 14 wide, and the direct match is the measurement. The rule, fixed in
[`kaggle_chess/build_1b_search_match.py`](kaggle_chess/build_1b_search_match.py) before
the games, changes the bot only if the interval excludes zero against it; it excludes
zero the other way, so the fine-tune's root stays. Against the 1B model at depth 1, the
searchless player of the paper: 901 won, 505 drawn, 194 lost, a score of 0.721 and
**+165 Elo (150 to 180)**, for 0.07 s per move at the median against 0.02. Search cost
the 500M model's fine-tune puzzles; in games, with the base model at the leaves, it is the
largest gain in this section.

**What a ply is worth** ([`results/34_match_500M_d3_vs_d2_part0.json`](results/34_match_500M_d3_vs_d2_part0.json)
and `part1`). The 500M model at depth 3 against itself at depth 2, on the same 200
positions played twice: 132 won, 153 drawn, 115 lost, a score of 0.521 and **+15 Elo (95%
interval -8 to 37)**. Not distinguishable from zero, and self-play tends to overstate what
a ply gives against other opponents. The clock is not what stops it: on a T4 sharing its
CPU cores with another job, depth 3 takes a median of 1.1 s per move, 3.1 s
at the 90th percentile and 12.7 s at worst, against 0.07, 0.23 and 2.0 s for depth 2, at
about 152 network calls per move against 2. It fits a blitz clock on a GPU and buys
little. The paper's 2054, blitz against bots with no search at all and about 41 times
these examples, stood 395 points above the bot's rapid rating then, in a different pool; a
ply of this size will not close that. In the paper's own category, blitz against bots,
the tree search below leaves the bot at a performance of 1784, about 270 below, and that
with search, which the paper's number does not use.

**A tree search instead of a fixed depth** ([`src/mcts.py`](src/mcts.py),
[`results/39_match_mcts32_vs_bot_part0.json`](results/39_match_mcts32_vs_bot_part0.json)
and `part1`). Fixed depth spends the same effort on every line; a tree search spends it
where the network is unsure. No random rollouts: each expansion is one network call
that returns Q for every legal move of the new position, the prior is a softmax of
those values (temperature 0.05), an unvisited move is worth the network's own Q, and a
visited one the lower of Q and its backed-up mean, the same pessimism as the
alpha-beta's min. Selection is PUCT, the root is scored by the fine-tune and the rest of
the tree by the base model, as in the bot. With 32 simulations a move costs up to 33 calls (32.4 on average in the games) of
about 30 moves each, against the roughly 930 (position, move) pairs of depth 2. The
exploration constant was picked before any game, on the first 2,000 test puzzles: 1,693,
1,733, **1,736** and 1,718 solved at c = 0.03, 0.1, 0.3 and 1.0, against 1,584 for the
bot's depth 2 on the same puzzles (185 solved only by the tree search, 33 only by depth
2; the comparison favours the setting it chose). Then the games, against the bot on the
800 positions played twice: 797 won, 519 drawn, 284 lost, a score of 0.660 and **+115 Elo
(95% interval 101 to 130)**, clustering the two games of each position. The rule, fixed in
[`kaggle_chess/build_mcts_match.py`](kaggle_chess/build_mcts_match.py) before the games,
changes the bot only if the interval excludes zero in the tree search's favour, and it
does by a wide margin. The price is time on a GPU, where fewer moves per call batch
worse: 0.28 s per move at the median against 0.05 for depth 2 in one pass
([`results/39_tune_summary.json`](results/39_tune_summary.json)). Tuning and 1,600 games
took 6.0 hours on two T4s.

**How many simulations** ([`results/41_match_mcts64_vs_mcts32_part0.json`](results/41_match_mcts64_vs_mcts32_part0.json)
and `part1`). The same tree search with 64 simulations against itself with 32, both with
the bot's networks and c = 0.3, on the same list of positions: a game costs about twice
as much, so the session's deadline stopped each card between positions after 625 of the
800. 495 won, 458 drawn, 297 lost, a score of 0.579 and **+56 Elo (95% interval 42 to
70)**. The rule, fixed in
[`kaggle_chess/build_mcts_sims_match.py`](kaggle_chess/build_mcts_sims_match.py) before
the games, moves the bot to 64 only if the interval excludes zero in 64's favour; it
does. Doubling the budget bought 56 here; a third ply of fixed depth bought 15 for
about 76 times the network calls.

**Tuning again at 64** ([`results/45_tune64_summary.json`](results/45_tune64_summary.json),
[`results/45_match_mcts64_c0.3_tau0.1_vs_bot_part0.json`](results/45_match_mcts64_c0.3_tau0.1_vs_bot_part0.json)
and `part1`). c = 0.3 and the prior temperature 0.05 were chosen at 32 simulations or
before; with 64 the tree grows deeper, so both were tried again on the first 2,000 test
puzzles, c at 0.3, 0.6 and 1.0 and the temperature at 0.05 and 0.1: between 1,731 and
1,755 solved, the bot's own setting 1,751. Too close to rank on puzzles, so the best
other setting, c = 0.3 at temperature 0.1, played the bot on the same positions until
the session's deadline, 500 of the 800 twice: 318 won, 403 drawn, 279 lost, a score of
0.520 and **+14 Elo (95% interval -1.5 to 29)**. The rule, fixed in
[`kaggle_chess/build_mcts_tune64.py`](kaggle_chess/build_mcts_tune64.py) before the
games, changes the bot only if the interval excludes zero; it does not, and the bot keeps
0.05. Tuning and 1,000 games took 10.2 hours on two T4s.

**Another doubling** ([`results/46_match_mcts128_vs_mcts64_part0.json`](results/46_match_mcts128_vs_mcts64_part0.json)
and `part1`). 128 simulations against 64, same networks, c = 0.3 and temperature 0.05,
the same list of positions; a game costs twice one of results/41, and the deadline
stopped the cards after 383 of the 800 positions: 275 won, 293 drawn, 198 lost, a score of
0.550 and **+35 Elo (95% interval 18 to 52)**. The rule, fixed in
[`kaggle_chess/build_mcts_sims_match.py`](kaggle_chess/build_mcts_sims_match.py) before
the games, moves the bot to 128 when its clock allows only if the interval excludes zero;
it does. On a T4, 128 took 1.15 s per move at the median against 0.58 for 64, exactly
twice, so about 7.7 s on the Kaggle CPU the bot plays on: the bot runs 128 with three
minutes or more left and below that the search of results/43-44. Doublings so far: +56
from 32 to 64, +35 from 64 to 128, intervals that overlap. 11.4 hours on two T4s.

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
leave an interval 300 Elo wide. `--leaf_ckpt` gives the bot the hybrid, now built on the
1B model (`--ckpt checkpoints/chess_1B_w03.pt --leaf_ckpt checkpoints/chess_1B.pt`).
That pairing played 66 rated rapid games (10+3) against 48 other bots on 18 November 2026,
in a Kaggle CPU session ([`kaggle_chess/build_lichess_bot.py`](kaggle_chess/build_lichess_bot.py),
the token from a Kaggle secret, the bot stopped before the 12-hour limit with no game
abandoned): 34 won, 4 drawn, 28 lost, the rating from 1659 to **1781**, a performance of
**1804 (1720 to 1892)** against opponents averaging 1773
([`results/38_lichess_1B_hybrid_session.json`](results/38_lichess_1B_hybrid_session.json)).
After the first 20 games the rating only moved between 1721 and 1785, against stronger
opponents as it rose; more games would narrow the interval and change no decision.
Since results/39 and 41 the bot plays the tree search instead, with 64 simulations
(`--sims 64 --c_puct 0.3`, same two networks). On the Kaggle CPU it plays on, over the
first 100 match positions, 32 simulations took 2.5 s per move at the median and 3.2 s at
worst against 2.9 and 5.0 s for depth 2, with the same move in 83 of them
([`results/40_bot_timing_cpu.json`](results/40_bot_timing_cpu.json)); in a second
session 64 simulations took 3.8 and 4.8 s against 2.0 and 2.5 s for 32 on that machine
([`results/42_bot_timing_cpu_mcts64.json`](results/42_bot_timing_cpu_mcts64.json),
[`kaggle_chess/build_bot_timing.py`](kaggle_chess/build_bot_timing.py)). Within a 10+3
clock either way. With the tree search the bot then played 43 rated rapid games against
31 other bots, from 20:55 on 25 November to 07:13 the next morning, in the same kind of
session: 19 won, 12 drawn, 12 lost, a score of 0.581, the rating from 1781 to **1855**, a
performance of **1933 (1830 to 2048)** against opponents averaging 1876, about 100 above
those of results/38
([`results/43_lichess_mcts64_session.json`](results/43_lichess_mcts64_session.json)).
After the first 20 games the rating stayed between 1824 and 1858. The two performance
intervals overlap, so online alone does not separate the searches; the 1,600 games of
results/39 and 41 do, and the session agrees with them. Search depth
adapts to the clock -- the bot declines time controls it cannot play at depth 2 and
drops to the bare network when short on time; the tree search first halves its
simulations under 90 s, which on results/43's own move times keeps a 3+2 game off the bare
network entirely. `--clocks "5+3,3+2"` makes a session of one speed, the one the paper's
number is in: it offers only those clocks and declines every other speed. One such
session, from 22:04 on 26 November to 09:00 the next morning with the same bot and
settings, played 60 rated games against 35 bots: 23 won, 8 drawn, 29 lost, a score of
0.45 and a performance of **1784 (1691 to 1872)** against opponents averaging 1819; the
blitz rating, provisional before the first game, ended at 1782
([`results/44_lichess_mcts64_blitz_session.json`](results/44_lichess_mcts64_blitz_session.json)).
19 more games were aborted before any move, which Lichess does not rate; they are
listed and not counted. The clock was not the cause: it never fell below 89 s, and no
move fell back on the bare network. The same bot performs about 150 lower in blitz
than in rapid against opponents of similar ratings, with intervals that overlap (1691 to
1872 and 1830 to 2048); why is not measured here. The comparison with the paper is under
[Distance to the paper](README.md#distance-to-the-paper). The API token comes from `LICHESS_TOKEN`
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

### One position, many moves

Choosing a move means valuing every legal move of the position, about 27 in the
middle of a game. Done as 27 separate inputs, that is 27 x 78 tokens through the
network. Attention is causal and the move is the last token, so the 77 position tokens
never see the move: their hidden states are the same for every move.
[`src/fast_eval.py`](src/fast_eval.py) runs the position once, keeps each layer's keys
and values, and then runs every move as a single token that attends to them, 77 + 27
tokens in all. This is the KV-cache of text generation, used for breadth instead of
length. It is exact: the logits match the plain forward to float rounding and the
chosen moves are the same (`tests/test_fast_eval.py`). The bot and the search use it.

Milliseconds per position for the 500M checkpoint on two CPUs, 4 threads each: an
Apple-silicon Mac (20 positions from random play, 26.8 legal moves on average) and a
Kaggle CPU session, x86 (50 positions, 29.0 moves;
[`results/31_int8_cpu_session.log`](results/31_int8_cpu_session.log)):

| | Mac, ARM | speed-up | Kaggle, x86 | speed-up |
|---|---|---|---|---|
| fp32, plain forward | 60.1 | 1.0x | 559.1 | 1.0x |
| fp32, shared prefix | **9.6** | **6.3x** | **69.7** | **8.0x** |
| int8 Linear layers, plain | 186.0 | 0.3x | 436.5 | 1.3x |
| int8 Linear layers, shared prefix | 15.3 | 3.9x | 54.8 | 10.2x |

The shared prefix gives the same answers: on Kaggle's CPU it solved the same 6,484 of
the 10,000 puzzles as the plain forward on a GPU (results/24). int8 dynamic
quantization is slower than fp32 on ARM, where PyTorch falls back to the qnnpack
kernels, and 1.27x faster on x86 with fbgemm. It is not free:
**63.75% of the puzzles against 64.84%**, 161 solved only by int8 and 270 only by
fp32, -1.09 points with a standard error of 0.21
([`results/31_puzzles_500M_int8_shared_cpu.json`](results/31_puzzles_500M_int8_shared_cpu.json),
paired with the fp32 file next to it). End to end, with the board logic included, the
10,000 puzzles took 1,045 s instead of 1,203 s. A point of accuracy for 13% of wall
clock is a bad trade for the bot, so it stays in fp32 with the prefix shared. ONNX was
not tried: the 6-8x comes from running fewer tokens, and a different runtime would
not change that.

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
