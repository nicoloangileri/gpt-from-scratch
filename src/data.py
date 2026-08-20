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


def prepare(text_path: str | Path, out_dir: str | Path, val_frac: float = 0.0005) -> dict:
    """Encode a raw text file to uint16 token .bin files (train/val).

    Run once per corpus; training then memory-maps the .bin files.
    """
    enc = get_tokenizer()
    text = Path(text_path).read_text(encoding="utf-8")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    n_val = max(1, int(len(ids) * val_frac))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids[:-n_val].tofile(out / "train.bin")
    ids[-n_val:].tofile(out / "val.bin")
    meta = {"vocab_size": enc.n_vocab, "n_train": len(ids) - n_val, "n_val": n_val}
    print(f"[data] {meta}")
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
