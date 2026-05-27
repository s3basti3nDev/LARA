"""
LARA — Local Evaluation Suite (no internet required).

Runs three evaluation classes:
  1. Perplexity — on held-out val set (bit-exact, reproducible)
  2. Throughput  — tokens/sec at batch_size=1 (inference speed)
  3. KV-cache   — theoretical compression ratio (MLA models)
  4. Depth scaling — PPL vs n_recursions at inference (RecurrentDepth)

Loads checkpoints saved by train_cpu.py and train_cpu_v2.py.

Usage:
    python evaluate.py                        # all available checkpoints
    python evaluate.py --model results/B_*.pt # specific checkpoint
    python evaluate.py --depth-scaling        # test RecurrentDepth at 1x,2x,4x
"""
import sys, os, math, time, json, argparse, glob
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from config import ModelConfig
from model.baseline import GPT
from model.lara import LARA
from model.mla import MultiHeadLatentAttention
from data.dataset import get_dataloaders


# ── Data ─────────────────────────────────────────────────────────
def _get_val_loader(block_size=128, batch_size=8, dataset="local"):
    tc = type("TC", (), {
        "dataset": dataset, "data_dir": "data",
        "block_size": block_size, "batch_size": batch_size,
    })()
    _, val_loader = get_dataloaders(tc)
    return val_loader


# ── Loaders ──────────────────────────────────────────────────────
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    extra_cfg = ckpt.get("extra_cfg", {})
    history = ckpt.get("history", [])
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    mc = ModelConfig(**{
        "n_embd": 256, "n_head": 4, "n_layer": 6,
        "block_size": 128, "memory_size": 128,
        "n_recursions": 3, "mor_top_k": 0.5, "memory_lr": 0.01,
        "n_thinking_steps": 4, "coconut_ramp_steps": 3000,
        "coconut_start_iter": 9999,
        **extra_cfg,
    })

    cls = GPT if not any(extra_cfg.get(k) for k in
                         ["use_diff_attention", "use_mla", "use_mor",
                          "use_recurrent_depth", "use_titans", "use_coconut"]) \
              else LARA
    model = cls(mc)

    state = ckpt["model_state"]
    # Strip torch.compile() prefix (_orig_mod.) added when saving compiled models
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    # Migrate old single-projection memory (q_proj) → MFE-PA multi-projection (q_projs)
    if "memory.q_proj.weight" in state and not any(
            k.startswith("memory.q_projs") for k in state):
        w = state.pop("memory.q_proj.weight")
        if hasattr(model, "memory") and model.memory is not None:
            for i in range(model.memory.n_proj):
                state[f"memory.q_projs.{i}.weight"] = w.clone()

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, mc, history, best_val_loss


# ── Perplexity ───────────────────────────────────────────────────
@torch.no_grad()
def eval_perplexity(model, val_loader, n_batches=100):
    is_lara = isinstance(model, LARA)
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        out = model(x, y)
        losses.append(out[1].item())
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


# ── Throughput ───────────────────────────────────────────────────
@torch.no_grad()
def eval_throughput(model, block_size=128, batch_size=1, n_steps=100):
    x = torch.randint(0, 50257, (batch_size, block_size))
    # warmup
    for _ in range(10):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        model(x)
    elapsed = time.perf_counter() - t0
    return (n_steps * batch_size * block_size) / elapsed


# ── Depth scaling ─────────────────────────────────────────────────
@torch.no_grad()
def eval_depth_scaling(model, val_loader, depths, n_batches=50):
    if not isinstance(model, LARA) or not model.config.use_recurrent_depth:
        return None

    results = {}
    for d in depths:
        # Reset Titans memory so each depth evaluation starts from a clean state
        if hasattr(model, "memory") and model.memory is not None:
            model.memory.reset_memory()
        losses = []
        for i, (x, y) in enumerate(val_loader):
            if i >= n_batches:
                break
            out = model(x, y, n_recursions_override=d)
            losses.append(out[1].item())
        results[d] = math.exp(sum(losses) / len(losses))
    return results


# ── KV cache info ─────────────────────────────────────────────────
def kv_info(model):
    for m in model.modules():
        if isinstance(m, MultiHeadLatentAttention):
            return {
                "type": "MLA",
                "d_c": m.d_c,
                "n_embd": m.n_embd,
                "compression_ratio": m.kv_cache_compression_ratio,
            }
    return {"type": "standard", "compression_ratio": 1.0}


# ── Main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=None,
                   help="Checkpoint path(s). Default: all results/*_ckpt.pt")
    p.add_argument("--depth-scaling", action="store_true",
                   help="Evaluate PPL at 1x,2x,3x recursions (RecurrentDepth only)")
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--dataset", default="local",
                   help="Dataset for evaluation: local|shakespeare|fineweb")
    args = p.parse_args()

    # Discover checkpoints
    if args.model:
        paths = args.model
    else:
        paths = sorted(set(glob.glob("results/*_ckpt.pt") + glob.glob("results/*_v2_ckpt.pt")))

    if not paths:
        print("No checkpoints found. Run train_cpu.py or train_cpu_v2.py first.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  LARA Evaluation Suite")
    print(f"{'='*80}")
    print(f"  {'Checkpoint':<35} {'Params':>8} {'Val PPL':>9} {'Tok/s':>9} {'KV cache':>10}")
    print(f"  {'-'*76}")

    depth_results = {}
    for path in paths:
        name = os.path.basename(path).replace("_ckpt.pt", "").replace("_v2_ckpt.pt", " v2")
        try:
            model, mc, history, best_val_loss = load_checkpoint(path)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue

        val_loader = _get_val_loader(block_size=mc.block_size, dataset=args.dataset)
        params = model.num_params()
        ppl = eval_perplexity(model, val_loader, args.n_batches)
        tps = eval_throughput(model, block_size=mc.block_size)
        kv  = kv_info(model)
        kv_str = f"{kv['compression_ratio']:.0f}x ({kv['type']})"

        print(f"  {name:<35} {params:>8,} {ppl:>9.2f} {tps:>9,.0f} {kv_str:>10}")

        if args.depth_scaling:
            d_r = eval_depth_scaling(model, val_loader, [1, 2, 3, 6, 9], args.n_batches // 2)
            if d_r:
                depth_results[name] = d_r

    if depth_results:
        print(f"\n  {'-- Depth Scaling (RecurrentDepth models) --':^76}")
        print(f"  {'Checkpoint':<35}", end="")
        depths_tested = sorted(next(iter(depth_results.values())).keys())
        for d in depths_tested:
            print(f"  {'R='+str(d):>6}", end="")
        print()
        for name, dr in depth_results.items():
            print(f"  {name:<35}", end="")
            for d in depths_tested:
                print(f"  {dr.get(d, float('nan')):>6.2f}", end="")
            print()

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
