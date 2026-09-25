#!/bin/bash
# Launch the queued Kaggle jobs once the weekly GPU quota resets, one after the other.
#
#   nohup kaggle_chess/queue_after_reset.sh >/dev/null 2>&1 &     # progress in kaggle_queue.log
#
# 1. the light puzzle fine-tune (rank weight 0.3), then
# 2. the 400-game match at depth 2 between the 225M model and the first fine-tune.
# A push refused for quota is retried every hour. The jobs themselves run on Kaggle;
# the laptop only has to be awake to start them and to fetch the results, and each
# result lands under a NEW name -- nothing already on disk is overwritten.
set -u
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
LOG=kaggle_queue.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

push_when_quota() {                     # $1 = notebook folder
    while :; do
        out=$($K kernels push -p "$1" 2>&1 | tail -1)
        if echo "$out" | grep -q "successfully pushed"; then say "$1: $out"; return; fi
        say "$1: not started ($out) -- retrying in 1h"
        sleep 3600
    done
}

wait_done() {                           # $1 = kernel id
    while :; do
        s=$($K kernels status "$1" 2>&1 | tail -1)
        case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) say "$1: $s"; return;; esac
        sleep 600
    done
}

fetch() {                               # $1 = kernel id, $2 = local folder
    $K kernels output "$1" -p "$2" --file-pattern '(chess_best\.pt|results\.json|\.log)$' -q
    say "$1: output in $2/"
}

say "queue started"
push_when_quota kaggle_chess_puzzle_w03
wait_done nicolangileri/gpt-from-scratch-chess-puzzle-w03
fetch nicolangileri/gpt-from-scratch-chess-puzzle-w03 kaggle_w03_out
if [ -f kaggle_w03_out/checkpoints/chess_best.pt ] && [ ! -e checkpoints/chess_225M_puzzles_w0.3.pt ]; then
    cp kaggle_w03_out/checkpoints/chess_best.pt checkpoints/chess_225M_puzzles_w0.3.pt
    say "saved checkpoints/chess_225M_puzzles_w0.3.pt"
fi

push_when_quota kaggle_chess_match_d2
wait_done nicolangileri/gpt-from-scratch-chess-match-d2
fetch nicolangileri/gpt-from-scratch-chess-match-d2 kaggle_match_d2_out
say "queue done"
