"""Tokenization + data loading.

Level 1 uses the GPT-2 BPE tokenizer via `tiktoken` (fast, battle-tested) and a
memory-mapped token stream so we can train on corpora larger than RAM. Point
`--data` at any UTF-8 text file (start with TinyStories or a FineWeb-edu shard).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_tokenizer(name: str = "gpt2"):
    import tiktoken
    return tiktoken.get_encoding(name)


def prepare(
    text_path: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.0005,
    val_tokens: int | None = None,
    chunk_bytes: int = 32 << 20,
) -> dict:
    """Encode a raw text file to uint16 token .bin files (train/val).

    Small corpora are tokenized in one shot. Anything above `chunk_bytes` (or any
    call that pins `val_tokens`) goes through the streaming path: read the file in
    line-aligned blocks, tokenize each block with tiktoken's parallel batch API and
    append straight to disk, so a multi-GB corpus never has to fit in RAM.

    Run once per corpus; training then memory-maps the .bin files.
    """
    enc = get_tokenizer()
    src = Path(text_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if val_tokens is None and src.stat().st_size <= chunk_bytes:
        ids = np.array(enc.encode_ordinary(src.read_text(encoding="utf-8")), dtype=np.uint16)
        # val must hold enough tokens for a batch at any reasonable block_size; the
        # default frac is tuned for huge corpora, so floor it (and cap on tiny ones).
        n_val = min(len(ids) // 4, max(4096, int(len(ids) * val_frac)))
        ids[:-n_val].tofile(out / "train.bin")
        ids[-n_val:].tofile(out / "val.bin")
        meta = {"vocab_size": enc.n_vocab, "n_train": len(ids) - n_val, "n_val": n_val}
        print(f"[data] {meta}")
        return meta

    # streaming path: val is filled first (held-out head), the rest becomes train
    target_val = val_tokens if val_tokens is not None else max(4096, int(src.stat().st_size * val_frac / 4))
    n_val = n_train = 0
    with open(src, "r", encoding="utf-8", errors="ignore") as fh, \
            open(out / "val.bin", "wb") as fval, open(out / "train.bin", "wb") as ftrain:
        while True:
            lines = fh.readlines(chunk_bytes)
            if not lines:
                break
            ids = np.fromiter(
                (t for piece in enc.encode_ordinary_batch(lines, num_threads=8) for t in piece),
                dtype=np.uint16,
            )
            if n_val < target_val:
                take = min(target_val - n_val, len(ids))
                ids[:take].tofile(fval)
                n_val += take
                ids = ids[take:]
            if len(ids):
                ids.tofile(ftrain)
                n_train += len(ids)
            print(f"[data] {n_train + n_val:,} tokens encoded", end="\r", flush=True)

    meta = {"vocab_size": enc.n_vocab, "n_train": n_train, "n_val": n_val}
    print(f"\n[data] {meta}")
    return meta


class TokenStream:
    """Memory-mapped token .bin; yields random (x, y) batches of length block_size."""

    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([self._slice(i) for i in ix])
        y = torch.stack([self._slice(i + 1) for i in ix])
        if device.startswith("cuda"):
            # pinned + non_blocking async copy to GPU
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def _slice(self, i: int) -> torch.Tensor:
        chunk = self.data[i : i + self.block_size].astype(np.int64)
        return torch.from_numpy(chunk)
