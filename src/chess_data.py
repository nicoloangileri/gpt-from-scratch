"""ChessBench (DeepMind) ingestion: .bag reader, FEN tokenizer, move vocabulary.

The corpus is `action_value` data: one record per (position, legal move) pair,
annotated by Stockfish 16 with the win probability of playing that move. We
group the records of a position and keep the argmax move, which turns the data
into a policy target -- one forward pass per move at play time, instead of one
per legal move.

Record layout (Apache Beam TupleCoder, decoded here without the dependency):

    varint(len) fen_utf8  varint(len) move_utf8  float64_be(win_prob)

The trailing float is fixed width, so a shard can be parsed straight through
without the file's record index -- which matters because the index lives at the
end of the file and the full train split is 1.1 TB across 2148 shards.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

# --- FEN tokenizer (77 tokens, 31-symbol vocabulary) -------------------------
# Same scheme as DeepMind's, so results stay comparable with the paper.
_CHARACTERS = list("0123456789abcdefghpnrkqPBNRQKw.")
_CHARACTERS_INDEX = {c: i for i, c in enumerate(_CHARACTERS)}
_SPACES = frozenset("12345678")
SEQUENCE_LENGTH = 77
VOCAB_SIZE = len(_CHARACTERS)


def tokenize_fen(fen: str) -> np.ndarray:
    """FEN -> fixed-length uint8 array of 77 tokens."""
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")
    board = side + board.replace("/", "")

    idx = []
    for char in board:
        if char in _SPACES:
            idx.extend(int(char) * [_CHARACTERS_INDEX["."]])
        else:
            idx.append(_CHARACTERS_INDEX[char])

    # castling padded to 4, en passant to 2, clocks to 3 digits each
    castling = "" if castling == "-" else castling
    idx.extend(_CHARACTERS_INDEX[c] for c in castling.ljust(4, "."))
    en_passant = "" if en_passant == "-" else en_passant
    idx.extend(_CHARACTERS_INDEX[c] for c in en_passant.ljust(2, "."))
    idx.extend(_CHARACTERS_INDEX[c] for c in halfmoves.ljust(3, "."))
    idx.extend(_CHARACTERS_INDEX[c] for c in fullmoves.ljust(3, "."))

    assert len(idx) == SEQUENCE_LENGTH, f"{len(idx)} tokens for {fen!r}"
    return np.asarray(idx, dtype=np.uint8)


# --- move vocabulary (1968 actions) ------------------------------------------
def _all_possible_moves() -> list[str]:
    """Every UCI move that could ever be legal, in DeepMind's generation order."""
    import chess

    moves = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        targets = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        targets += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        targets += board.attacks(square)
        board.remove_piece_at(square)
        moves += [chess.square_name(square) + chess.square_name(t) for t in targets]

    files = "abcdefgh"
    for rank, next_rank in [("2", "1"), ("7", "8")]:
        for i, file in enumerate(files):
            squares = [file]                                  # straight push
            if file > "a":
                squares.append(files[i - 1])                  # capture left
            if file < "h":
                squares.append(files[i + 1])                  # capture right
            for next_file in squares:
                base = f"{file}{rank}{next_file}{next_rank}"
                moves += [base + p for p in "qrbn"]
    return moves


MOVES = _all_possible_moves()
MOVE_TO_ACTION = {m: i for i, m in enumerate(MOVES)}
ACTION_TO_MOVE = {i: m for i, m in enumerate(MOVES)}
NUM_ACTIONS = len(MOVES)


# --- .bag reading -------------------------------------------------------------
def _read_varint(buf, pos: int) -> tuple[int, int]:
    """Beam's varint: 7 bits per byte, little-endian, high bit = continue."""
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def iter_action_value(path: str | Path, limit: int | None = None):
    """Yield (fen, move, win_prob) straight from a shard, index-free."""
    with open(path, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        # The record index is appended after the records; stop before it.
        (index_start,) = struct.unpack("<Q", buf[-8:])
        pos, n = 0, 0
        while pos < index_start:
            length, pos = _read_varint(buf, pos)
            fen = buf[pos : pos + length].decode()
            pos += length
            length, pos = _read_varint(buf, pos)
            move = buf[pos : pos + length].decode()
            pos += length
            (win_prob,) = struct.unpack(">d", buf[pos : pos + 8])
            pos += 8
            yield fen, move, win_prob
            n += 1
            if limit is not None and n >= limit:
                return
    finally:
        buf.close()


# --- packing into training tensors -------------------------------------------
# A sample is [77 FEN tokens] + [1 action token]; the target is the Stockfish
# win probability discretised into `num_buckets` uniform bins. Classification
# over bins beats scalar regression here (Ruoss et al., 2024), and it keeps the
# training loop identical to the language-model one: cross-entropy on logits.
ACTION_OFFSET = VOCAB_SIZE                      # actions live after the FEN symbols
FULL_VOCAB = VOCAB_SIZE + NUM_ACTIONS           # 31 + 1968 = 1999
SAMPLE_LENGTH = SEQUENCE_LENGTH + 1             # 78


def win_prob_to_bucket(win_prob: np.ndarray, num_buckets: int) -> np.ndarray:
    """Uniform bins over [0, 1]; 0.0 -> bin 0, 1.0 -> last bin."""
    edges = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    return np.searchsorted(edges, win_prob, side="left").astype(np.uint8)


def prepare_chessbench(
    shards: list[str | Path],
    out_dir: str | Path,
    num_buckets: int = 64,
    max_records: int | None = None,
    val_records: int = 200_000,
    report_every: int = 1_000_000,
) -> dict:
    """Stream shards into states/actions/buckets .bin files (val filled first)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {
        split: {name: open(out / f"{split}_{name}.bin", "wb")
                for name in ("states", "actions", "buckets")}
        for split in ("val", "train")
    }
    counts = {"val": 0, "train": 0}
    skipped = 0

    def flush(split, states, actions, probs):
        handles[split]["states"].write(np.stack(states).tobytes())
        handles[split]["actions"].write(np.asarray(actions, dtype=np.uint16).tobytes())
        handles[split]["buckets"].write(
            win_prob_to_bucket(np.asarray(probs, dtype=np.float64), num_buckets).tobytes())
        counts[split] += len(states)

    buf_s, buf_a, buf_p = [], [], []
    total = 0
    try:
        for shard in shards:
            for fen, move, win_prob in iter_action_value(shard):
                action = MOVE_TO_ACTION.get(move)
                if action is None:          # should not happen, but never crash a run
                    skipped += 1
                    continue
                buf_s.append(tokenize_fen(fen))
                buf_a.append(action)
                buf_p.append(win_prob)
                total += 1
                if len(buf_s) >= 100_000:
                    split = "val" if counts["val"] < val_records else "train"
                    flush(split, buf_s, buf_a, buf_p)
                    buf_s, buf_a, buf_p = [], [], []
                    if total % report_every < 100_000:
                        print(f"[chess] {total:,} records", flush=True)
                if max_records is not None and total >= max_records:
                    raise StopIteration
    except StopIteration:
        pass
    if buf_s:
        flush("val" if counts["val"] < val_records else "train", buf_s, buf_a, buf_p)
    for split in handles.values():
        for fh in split.values():
            fh.close()

    meta = {"train": counts["train"], "val": counts["val"], "skipped": skipped,
            "num_buckets": num_buckets, "vocab": FULL_VOCAB}
    print(f"[chess] {meta}")
    return meta
