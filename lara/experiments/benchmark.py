"""
LARA Phase 1 — Comparative Benchmark

Loads checkpoints for all 5 experiments and reports:
  - Validation perplexity
  - Trainable parameters
  - FLOPs/token estimate
  - Inference throughput (tokens/sec)
  - GPU memory peak

Usage:
    python experiments/benchmark.py
    python experiments/benchmark.py --device cpu
"""
import os
import sys
import time
import math
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    baseline_config, diff_attn_config, mor_config,
    coconut_config, lara_full_config, lara_v2_config, lara_v2_full_config,
)
from model.baseline import GPT
from model.lara import LARA
from model.mla import MultiHeadLatentAttention
from data.dataset import get_dataloaders


# Results checkpoints saved by train_cpu.py / train_cpu_v2.py
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EXPERIMENTS = {
    "A -- Baseline GPT":       (baseline_config,     GPT,  "A_--_Baseline_GPT____ckpt.pt",  False),
    "B -- LARA v1-3":          (mor_config,          LARA, "B_--_LARA_v1-3______v2_ckpt.pt", True),
    "D -- LARA v2":            (lara_v2_config,      LARA, "D_--_LARA_v2________v2_ckpt.pt", True),
    "E -- LARA v2+Coco":       (lara_v2_full_config, LARA, "E_--_LARA_v2+Coco___v2_ckpt.pt", True),
}

CKPT_DIR = RESULTS_DIR


def load_model(cfg_fn, model_cls, ckpt_name, is_lara, device):
    mc, tc = cfg_fn()
    model = model_cls(mc).to(device)

    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    val_loss = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt.get("model", {})))
        val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss"))

    kv_ratio = None
    if is_lara:
        for m in model.modules():
            if isinstance(m, MultiHeadLatentAttention):
                kv_ratio = m.kv_cache_compression_ratio
                break

    return model, tc, val_loss, kv_ratio


@torch.no_grad()
def throughput(model, device, is_lara, block_size=512, batch_size=4, n_steps=50):
    model.eval()
    x = torch.randint(0, 1000, (batch_size, block_size), device=device)
    # Warmup
    for _ in range(5):
        if is_lara:
            model(x)
        else:
            model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None

    t0 = time.time()
    for _ in range(n_steps):
        if is_lara:
            model(x)
        else:
            model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    elapsed = time.time() - t0
    tokens_per_sec = (n_steps * batch_size * block_size) / elapsed
    return tokens_per_sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    print(f"\n{'='*80}")
    print(f"  LARA Benchmark  ({device})")
    print(f"{'='*80}")
    header = (f"{'Experiment':<22} {'Params':>10} {'Val PPL':>9} "
              f"{'Tok/s':>9} {'KV ratio':>9} {'FLOPs/tok':>12}")
    print(header)
    print("-" * 80)

    for name, (cfg_fn, model_cls, ckpt, is_lara) in EXPERIMENTS.items():
        model, tc, val_loss, kv_ratio = load_model(cfg_fn, model_cls, ckpt, is_lara, device)

        params = model.num_params()
        ppl = math.exp(val_loss) if val_loss is not None else float("nan")
        tps = throughput(model, device, is_lara)
        flops = model.flops_per_token_estimate() if hasattr(model, "flops_per_token_estimate") \
                else 6 * params
        kv_str = f"{kv_ratio:.0f}x" if kv_ratio else "1x"

        trained = "" if val_loss is not None else "  *"
        print(f"{name:<22} {params:>10,} {ppl:>9.2f} {tps:>9,.0f} "
              f"{kv_str:>9} {flops:>12,.0f}{trained}")

    print("=" * 80)
    print("\nNotes:")
    print("  Val PPL  — lower is better (perplexity on validation set)")
    print("  Tok/s    — inference throughput (higher is better)")
    print("  KV ratio — KV-cache size vs standard MHA (higher = more efficient)")
    print("  FLOPs/tok — 6*N estimate  *not trained yet")


if __name__ == "__main__":
    main()
