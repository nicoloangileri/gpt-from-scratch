"""Search on puzzles, paired by puzzle, and the session that runs it next to a match.

compare_search is a measurement that runs for hours on a quota that cannot be topped
up, so what is checked here is what would waste it silently: a configuration scored
on different puzzles than another, a paired difference with the sign flipped, a
deadline that leaves nothing on disk, and a notebook that picks up the wrong
checkpoint because two kernels both wrote a chess_best.pt.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import chess
import pytest
import torch

from src.chess_data import FULL_VOCAB, SAMPLE_LENGTH
from src.model import GPT, GPTConfig
from src.puzzles_eval import compare_search

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_chess"))
import build_two_gpus as b  # noqa: E402

# the two fixture puzzles of tests/test_chess_data.py
PUZZLES = [("aaaaa", 1200, "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35",
            "e5f6 e8e1 g1f2 e1f1"),
           ("bbbbb", 700, "6k1/5ppp/8/8/8/8/8/RR4K1 b - - 0 1", "g8h8 a1a8")]


def _csv(path: Path, times: int = 1) -> Path:
    lines = ["PuzzleId,Rating,FEN,Moves"]
    for i in range(times):
        lines += [f"{pid}{i},{rating},{fen},{moves}" for pid, rating, fen, moves in PUZZLES]
    path.write_text("\n".join(lines) + "\n")
    return path


def _oracle():
    answer = {}
    for _, _, fen, moves in PUZZLES:
        board = chess.Board(fen)
        for ply, uci in enumerate(moves.split()):
            if ply % 2 == 1:
                answer[board.fen()] = uci
            board.push(chess.Move.from_uci(uci))
    return lambda board: answer[board.fen()]


def _wrong(board):
    right = _oracle()(board)
    return next(m.uci() for m in board.legal_moves
                if m.uci() != right and not _mates(board, m))


def _mates(board, move):
    b = board.copy()
    b.push(move)
    return b.is_checkmate()


def test_pairs_are_scored_on_the_same_puzzles_with_the_right_sign(tmp_path):
    picks = {"m@1": _wrong, "m@2": _oracle(), "n@1": _oracle(), "n@2": _oracle()}
    out = tmp_path / "c.json"
    s = compare_search({}, str(_csv(tmp_path / "p.csv", times=3)), 5, "cpu",
                       out=str(out), picks=picks)
    assert s["attempted"] == 5                                   # --num caps the list
    assert s["all"]["m@1"]["solved"] == 0 and s["all"]["m@2"]["solved"] == 5
    d = s["depth2_minus_depth1"]
    assert d["m"]["points"] == 100 and d["m"]["only_depth2_solves"] == 5
    assert d["n"]["points"] == 0 and d["n"]["only_depth1_solves"] == 0
    saved = json.loads(out.read_text())
    assert saved["finished"] and len(saved["rows"]) == 5
    assert [r["id"] for r in saved["rows"]] == ["aaaaa0", "bbbbb0", "aaaaa1", "bbbbb1", "aaaaa2"]
    assert set(saved["all_by_rating"]) == {"500-999", "1000-1499"}


def test_a_deadline_already_past_still_leaves_a_result(tmp_path):
    out = tmp_path / "c.json"
    s = compare_search({}, str(_csv(tmp_path / "p.csv")), 10, "cpu", hours=0, out=str(out),
                       picks={"m@1": _oracle(), "m@2": _oracle()})
    assert s["attempted"] == 0 and json.loads(out.read_text())["finished"]


def test_real_checkpoints_at_both_depths(tmp_path):
    """The path the notebook takes: checkpoints on disk, run()'s pickers, depth-2 search."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=FULL_VOCAB, block_size=SAMPLE_LENGTH, n_layer=1, n_head=2,
                    n_embd=32, causal=False, num_classes=8)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"cfg": asdict(cfg), "model": GPT(cfg).state_dict(), "step": 0, "val": 0.0}, ckpt)
    s = compare_search({"a": str(ckpt), "b": str(ckpt)}, str(_csv(tmp_path / "p.csv")), 2, "cpu")
    assert s["attempted"] == 2
    assert s["all"]["a@1"] == s["all"]["b@1"] and s["all"]["a@2"] == s["all"]["b@2"]


def test_the_session_finds_each_checkpoint_by_the_kernel_that_wrote_it(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "setup_cell", lambda: "T0 = 0\n")
    out = b.build(6.7, tmp_path / "nb")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"       # two T4s
    assert meta["kernel_sources"] == [b.BASE_KERNEL, b.W03_KERNEL]
    assert meta["dataset_sources"] == [b.DATASET] and meta["id"] == b.KERNEL_ID
    cells = ["".join(c["source"]) for c in json.loads((out / "two_gpus_kernel.ipynb").read_text())["cells"]]
    assert "END = T0 + 6.62 * 3600" in cells[3]
    assert cells[3].count('"CUDA_VISIBLE_DEVICES": gpu') == 1 and '("0", ["-m", "src.match"' in cells[3]
    assert '("1", ["-m", "src.puzzles_eval"' in cells[3]

    inp = tmp_path / "input"                       # both kernels wrote a chess_best.pt
    for rel in ("notebooks/nicolangileri/gpt-from-scratch-chess-500m-s3/checkpoints/chess_best.pt",
                "notebooks/nicolangileri/gpt-from-scratch-chess-puzzle-500m-w03/checkpoints/chess_best.pt",
                "datasets/nicolangileri/gpt-from-scratch-checkpoints/chess_225M.pt"):
        (inp / rel).parent.mkdir(parents=True, exist_ok=True)
        (inp / rel).write_text(rel)
    ns = {}
    exec(cells[2].replace("/kaggle/input", str(inp)), ns)
    assert Path(ns["BASE500"]).read_text().split("/")[2] == "gpt-from-scratch-chess-500m-s3"
    assert Path(ns["W03"]).read_text().split("/")[2] == "gpt-from-scratch-chess-puzzle-500m-w03"
    assert ns["M225"].endswith("chess_225M.pt")
    with pytest.raises(SystemExit):
        b.build(0.5, tmp_path / "nb2")
