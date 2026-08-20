"""Generate text from a trained checkpoint.

    python -m src.sample --ckpt checkpoints/best.pt --prompt "Once upon a time" \
        --max_new_tokens 200 --temperature 0.8 --top_k 200
"""

from __future__ import annotations

import argparse

import torch

from .data import get_tokenizer
from .model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["cfg"]))
    # tolerate torch.compile's "_orig_mod." prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().to(args.device)

    enc = get_tokenizer()
    ids = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
