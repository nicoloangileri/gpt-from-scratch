"""Config-driven training loop for the from-scratch GPT.

    python -m src.train --config configs/config.yaml

Everything (model size, LR schedule, batch, steps, device) comes from the YAML +
CLI overrides, so runs are reproducible from one file + one command. Seeds are
fixed. Loss and throughput are logged each `log_interval`.

Level-2 hooks already wired: AMP autocast + GradScaler (mixed precision) and
gradient accumulation. Multi-GPU DDP and MFU logging are the next milestones —
see README.md.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .data import TokenStream
from .model import GPT, GPTConfig


def get_lr(step: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to min_lr."""
    warmup, total, lr, min_lr = cfg["warmup_steps"], cfg["max_steps"], cfg["lr"], cfg["min_lr"]
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, streams: dict, cfg: dict, ctx) -> dict:
    out = {}
    model.eval()
    for split, stream in streams.items():
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = stream.batch(cfg["batch_size"], cfg["device"])
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data_dir", default=None, help="dir with train.bin/val.bin (overrides yaml)")
    args, overrides = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    # simple `--key value` overrides for sweeps/ablations
    for i in range(0, len(overrides) - 1, 2):
        key = overrides[i].lstrip("-")
        if key in cfg:
            cur, raw = cfg[key], overrides[i + 1]
            # bool must be parsed from the string: bool("false") is True in Python.
            cfg[key] = (raw.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(raw)

    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = cfg["device"]
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

    # mixed precision (LEVEL 2): bf16/fp16 autocast on CUDA, no-op elsewhere
    use_amp = cfg.get("amp", True) and device_type == "cuda"
    # torch.cuda.is_bf16_supported() counts emulation, so ask the card directly:
    # native bf16 starts at Ampere (sm_80).
    native_bf16 = use_amp and torch.cuda.get_device_capability()[0] >= 8
    if use_amp and cfg.get("dtype", "bf16") == "bf16" and not native_bf16:
        # Turing (T4) and older have no native bf16: autocast falls back to a slow
        # emulated path. fp16 + GradScaler is the right choice on those cards.
        print(f"[amp] bf16 unsupported on {torch.cuda.get_device_name()} -> using fp16")
        cfg["dtype"] = "fp16"
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("dtype", "bf16"))
    ctx = torch.autocast(device_type="cuda", dtype=ptdtype) if use_amp else _nullctx()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and cfg.get("dtype") == "fp16")

    data_dir = Path(cfg["data_dir"])
    streams = {
        "train": TokenStream(data_dir / "train.bin", cfg["block_size"]),
        "val": TokenStream(data_dir / "val.bin", cfg["block_size"]),
    }

    model_cfg = GPTConfig(**{k: cfg[k] for k in GPTConfig.__dataclass_fields__ if k in cfg})
    model = GPT(model_cfg).to(device)
    print(f"[model] {model.num_params()/1e6:.2f}M non-embedding params | {asdict(model_cfg)}")
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["lr"], (cfg["beta1"], cfg["beta2"])
    )
    if cfg.get("compile", False):
        model = torch.compile(model)  # LEVEL 2: kernel fusion via torch.compile

    accum = cfg.get("grad_accum_steps", 1)
    ckpt_dir = Path(cfg.get("out_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = t_log = time.time()
    s_log = 0

    for step in range(cfg["max_steps"] + 1):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, cfg)

        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, streams, cfg, ctx)
            dt = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {dt:.1f}s")
            if losses["val"] < best_val and step > 0:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "cfg": asdict(model_cfg),
                            "step": step, "val": best_val}, ckpt_dir / "best.pt")
            t_log, s_log = time.time(), step  # don't charge eval time to throughput

        # gradient accumulation: split the effective batch into `accum` micro-steps
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            x, y = streams["train"].batch(cfg["batch_size"], device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            scaler.scale(loss).backward()
        if cfg.get("grad_clip", 0):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["log_interval"] == 0 and step > 0:
            # LEVEL 2: throughput. MFU (vs the card's peak FLOPs) is the next one.
            toks = cfg["batch_size"] * cfg["block_size"] * accum
            if device_type == "cuda":
                # CUDA is async: without this the clock measures how long the CPU
                # took to QUEUE the step, not to run it (one A/B run reported 79k
                # samples/s and an MFU of 882%).
                torch.cuda.synchronize()
            now = time.time()
            # steps actually run since the last reset (an eval in between moves it)
            tps = toks * (step + 1 - s_log) / max(1e-6, now - t_log)
            t_log, s_log = now, step + 1
            print(f"  step {step} | loss {loss.item()*accum:.4f} | lr {get_lr(step, cfg):.2e} "
                  f"| {toks} tok/step | {tps:,.0f} tok/s")

    print(f"[done] best val loss {best_val:.4f} -> {ckpt_dir/'best.pt'}")


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
