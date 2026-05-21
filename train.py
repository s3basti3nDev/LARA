"""
LARA training loop — works for both GPT baseline and any LARA variant.

Usage:
    python train.py --experiment baseline
    python train.py --experiment diff_attn
    python train.py --experiment mor
    python train.py --experiment coconut
    python train.py --experiment lara_full

    python train.py --experiment lara_full --wandb --max_iters 10000
"""
import os
import sys
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    baseline_config, diff_attn_config, mor_config,
    coconut_config, lara_full_config, lara_v2_config, lara_v2_full_config,
    lara_v2_dca_config, lara_v2_rope_config,
    TrainConfig,
)
from model.baseline import GPT
from model.lara import LARA
from data.dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline":      (baseline_config,     GPT),
    "diff_attn":     (diff_attn_config,    LARA),
    "mor":           (mor_config,          LARA),
    "coconut":       (coconut_config,      LARA),
    "lara_full":     (lara_full_config,    LARA),
    "lara_v2":       (lara_v2_config,      LARA),  # MLA + RecurrentDepth + Titans
    "lara_v2_full":  (lara_v2_full_config, LARA),  # + Coconut (deferred)
    "lara_v2_dca":   (lara_v2_dca_config,  LARA),  # + DCA (Brique 2c)
    "lara_v2_rope":  (lara_v2_rope_config, LARA),  # + RoPE (Brique 5)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="baseline",
                   choices=list(EXPERIMENTS.keys()))
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dataset", default=None,
                   help="Override dataset (shakespeare|fineweb)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--n_recursions", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--warmup_iters", type=int, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# LR schedule — cosine with linear warmup
# ──────────────────────────────────────────────────────────────

def get_lr(it: int, tc: TrainConfig) -> float:
    if it < tc.warmup_iters:
        return tc.learning_rate * it / tc.warmup_iters
    if it > tc.lr_decay_iters:
        return tc.min_lr
    decay = (it - tc.warmup_iters) / (tc.lr_decay_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


# ──────────────────────────────────────────────────────────────
# Loss helper (handles both GPT and LARA return signatures)
# ──────────────────────────────────────────────────────────────

def forward_loss(model, x, y, is_lara: bool, iter_num: int = 0):
    if is_lara:
        _, loss, aux = model(x, y, iter_num=iter_num)
    else:
        _, loss = model(x, y)
        aux = {}
    return loss, aux


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, device, is_lara, eval_iters):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        loss, _ = forward_loss(model, x, y, is_lara)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    cfg_fn, model_cls = EXPERIMENTS[args.experiment]
    mc, tc = cfg_fn()

    if args.max_iters:   tc.max_iters = args.max_iters
    if args.wandb:       tc.wandb_log = True
    if args.device:      tc.device = args.device
    if args.compile:     tc.compile = True
    if args.dataset:     tc.dataset = args.dataset
    if args.batch_size:  tc.batch_size = args.batch_size
    if args.n_embd:      mc.n_embd = args.n_embd
    if args.n_layer:     mc.n_layer = args.n_layer
    if args.n_head:      mc.n_head = args.n_head
    if args.block_size:
        mc.block_size = args.block_size
        tc.block_size = args.block_size
    if args.eval_interval:
        tc.eval_interval = args.eval_interval
    if args.n_recursions:  mc.n_recursions = args.n_recursions
    if args.learning_rate: tc.learning_rate = args.learning_rate
    if args.warmup_iters:  tc.warmup_iters = args.warmup_iters
    grad_accum = args.grad_accum

    device = tc.device
    print(f"[LARA] Démarrage — experiment={args.experiment}  device={device}", flush=True)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", flush=True)
        device = "cpu"
    dtype = getattr(torch, tc.dtype) if tc.dtype != "float32" else torch.float32
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=dtype) \
          if device != "cpu" else nullcontext()

    is_lara = model_cls == LARA

    # Data
    print("[LARA] Chargement du dataset...", flush=True)
    train_loader, val_loader = get_dataloaders(tc)
    train_iter = iter(train_loader)
    print("[LARA] Dataset OK.", flush=True)

    # Model
    print("[LARA] Initialisation du modèle...", flush=True)
    model = model_cls(mc).to(device)
    if tc.compile and hasattr(torch, "compile"):
        print("[LARA] Compilation torch.compile() en cours (2-5 min)...", flush=True)
        model = torch.compile(model)
        print("[LARA] Compilation terminée.", flush=True)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        betas=(tc.beta1, tc.beta2),
        weight_decay=tc.weight_decay,
    )

    # Resume
    start_iter = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"[LARA] Reprise depuis {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter_num"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"[LARA] Reprise à iter {start_iter}  best_val_loss={best_val_loss:.4f}", flush=True)

    # W&B
    if tc.wandb_log:
        import wandb
        wandb.init(project=tc.wandb_project, name=tc.run_name, config={
            **vars(mc), **vars(tc)
        })

    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment : {args.experiment}", flush=True)
    print(f"  Model      : {model_cls.__name__}", flush=True)
    print(f"  Params     : {model.num_params():,}", flush=True)
    print(f"  Device     : {device}  |  dtype: {tc.dtype}", flush=True)
    print(f"  Max iters  : {tc.max_iters:,}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── Training loop ──────────────────────────────────────────
    t0 = time.time()
    raw_model = model.module if hasattr(model, "module") else model

    for iter_num in range(start_iter, tc.max_iters + 1):
        # LR update
        lr = get_lr(iter_num, tc)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval
        if iter_num % tc.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, is_lara, tc.eval_iters)
            model.train()  # restore train mode after eval
            ppl = math.exp(min(val_loss, 20))
            print(f"[{iter_num:6d}]  val_loss={val_loss:.4f}  "
                  f"ppl={ppl:.2f}  lr={lr:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")
            if tc.wandb_log:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": ppl,
                           "lr": lr}, step=iter_num)
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {
                "model_state":   raw_model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "iter_num":      iter_num,
                "best_val_loss": best_val_loss,
                "extra_cfg":     vars(mc),
                "train_config":  vars(tc),
            }
            torch.save(ckpt, f"checkpoints/{tc.run_name}_last.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt["best_val_loss"] = val_loss
                torch.save(ckpt, f"checkpoints/{tc.run_name}_best.pt")

        if iter_num == tc.max_iters:
            break

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                # Reset Titans memory at epoch boundary
                if is_lara and hasattr(raw_model, "memory") and raw_model.memory is not None:
                    raw_model.memory.reset_memory()
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with ctx:
                loss, aux = forward_loss(model, x, y, is_lara, iter_num)
            (loss / grad_accum).backward()
            accum_loss += loss.item() / grad_accum

        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        if iter_num % tc.log_interval == 0:
            t1 = time.time()
            aux_str = ""
            if aux.get("aux_loss") is not None:
                aux_str = f"  aux={aux['aux_loss'].item():.4f}"
            print(f"[{iter_num:6d}]  train_loss={accum_loss:.4f}{aux_str}  "
                  f"dt={(t1-t0)*1000/tc.log_interval:.0f}ms/iter")
            t0 = t1
            if tc.wandb_log:
                import wandb
                wandb.log({"train/loss": accum_loss, "lr": lr}, step=iter_num)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
