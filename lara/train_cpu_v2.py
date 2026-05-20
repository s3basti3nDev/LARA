"""
LARA Phase 2 — CPU benchmark: v1 briques vs v2 architecture.

Compares:
  A  Baseline GPT         (reference)
  B  LARA v1-3            DiffAttn + MoR + Titans  (Phase 1 winner)
  D  LARA v2              MLA + RecurrentDepth + Titans
  E  LARA v2 + Coconut    MLA + RecurrentDepth + Titans + Coconut (deferred)

Key differences v1 -> v2:
  - DiffAttn replaced by MLA: 16x smaller KV-cache, per-head RMSNorm
  - MoR (token-adaptive depth) replaced by RecurrentDepth (global, test-time scalable)
  - Coconut curriculum now uses coconut_start_iter to defer to fine-tune phase
"""
import sys, os, math, time, json
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from model.baseline import GPT
from model.lara import LARA
from data.dataset import get_dataloaders

# ── Config CPU-friendly ───────────────────────────────────────
CPU_MODEL = dict(
    n_embd       = 256,
    n_head       = 4,
    n_layer      = 6,
    block_size   = 128,
    dropout      = 0.0,
    bias         = False,
    memory_size  = 128,
    n_recursions = 3,
    mor_top_k    = 0.5,
    memory_lr    = 0.01,
    # Coconut deferred beyond max_iters so Phase 1 never activates it
    n_thinking_steps   = 4,
    coconut_ramp_steps = 3000,
    coconut_start_iter = 9999,   # effectively disabled in 1000-step Phase 1
)

CPU_TRAIN = dict(
    dataset    = "local",
    data_dir   = "data",
    batch_size = 6,
    block_size = 128,
    max_iters  = 1000,
    learning_rate = 3e-4,
    weight_decay  = 0.1,
    grad_clip     = 1.0,
    warmup_iters  = 100,
    lr_decay_iters = 1000,
    min_lr     = 3e-5,
    eval_interval = 100,
    eval_iters    = 50,
    log_interval  = 50,
    device = "cpu",
    dtype  = "float32",
)

EXPERIMENTS = [
    ("A -- Baseline GPT   ", GPT,  dict()),
    ("B -- LARA v1-3      ", LARA, dict(
        use_diff_attention = True,
        use_mor            = True,
        use_coconut        = False,
        use_titans         = True,
    )),
    ("D -- LARA v2        ", LARA, dict(
        use_mla              = True,
        use_recurrent_depth  = True,
        use_coconut          = False,
        use_titans           = True,
        n_recursions         = 2,      # FLOPs-matched to LARA-3 (MoR avg=2 rec)
    )),
    ("E -- LARA v2+Coco   ", LARA, dict(
        use_mla              = True,
        use_recurrent_depth  = True,
        use_coconut          = True,   # deferred via coconut_start_iter
        use_titans           = True,
        n_recursions         = 2,
    )),
]

# ── Helpers (identical to train_cpu.py) ──────────────────────
def get_lr(it, tc):
    if it < tc["warmup_iters"]:
        return tc["learning_rate"] * it / tc["warmup_iters"]
    if it > tc["lr_decay_iters"]:
        return tc["min_lr"]
    progress = (it - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
    return tc["min_lr"] + 0.5 * (1 + math.cos(math.pi * progress)) * (tc["learning_rate"] - tc["min_lr"])

@torch.no_grad()
def evaluate(model, loader, is_lara, n=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        out = model(x, y)
        losses.append(out[1].item())
    return sum(losses) / len(losses)

def make_model(cls, extra_cfg, seed=42):
    torch.manual_seed(seed)
    mc = ModelConfig(**{**CPU_MODEL, **extra_cfg})
    return cls(mc)

def run_experiment(name, cls, extra_cfg, train_loader, val_loader, seed=42):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    is_lara = cls == LARA
    model = make_model(cls, extra_cfg, seed=seed)
    n_params = model.num_params()
    print(f"  Parametres : {n_params:,}")

    if is_lara and hasattr(model.config, "use_mla") and model.config.use_mla:
        from model.mla import MultiHeadLatentAttention
        attn_mod = None
        for m in model.modules():
            if isinstance(m, MultiHeadLatentAttention):
                attn_mod = m
                break
        if attn_mod:
            ratio = attn_mod.kv_cache_compression_ratio
            print(f"  KV-cache reduction : {ratio:.1f}x (MLA)")

    tc = CPU_TRAIN
    opt = torch.optim.AdamW(model.parameters(),
                            lr=tc["learning_rate"],
                            weight_decay=tc["weight_decay"])

    train_iter = iter(train_loader)
    history = []
    best_val_loss = float("inf")
    best_ppl = float("inf")
    t_start = time.time()

    for it in range(tc["max_iters"] + 1):
        lr = get_lr(it, tc)
        for pg in opt.param_groups: pg["lr"] = lr

        if it % tc["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, is_lara, tc["eval_iters"])
            model.train()
            ppl = math.exp(val_loss)
            elapsed = time.time() - t_start
            flag = " *" if val_loss < best_val_loss else ""
            print(f"  [{it:5d}/{tc['max_iters']}]  val_loss={val_loss:.4f}  "
                  f"ppl={ppl:.2f}  lr={lr:.1e}  {elapsed:.0f}s{flag}")
            history.append({"iter": it, "val_loss": val_loss, "ppl": ppl})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ppl = ppl

        if it == tc["max_iters"]: break

        try:
            x, y = next(train_iter)
        except StopIteration:
            if is_lara and hasattr(model, "memory") and model.memory is not None:
                model.memory.reset_memory()
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        opt.zero_grad(set_to_none=True)
        out = model(x, y) if not is_lara else model(x, y, iter_num=it)
        loss = out[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()

        if it % tc["log_interval"] == 0 and it > 0:
            print(f"  [{it:5d}]  train_loss={loss.item():.4f}")

    print(f"  Meilleur val PPL : {best_ppl:.2f}")
    os.makedirs("results", exist_ok=True)
    safe_name = name.strip().replace(" ", "_").replace("--", "-")
    with open(f"results/{safe_name}_v2.json", "w") as f:
        json.dump({"name": name, "params": n_params, "history": history}, f, indent=2)

    if is_lara:
        ckpt_path = f"results/{safe_name}_v2_ckpt.pt"
        torch.save({
            "model_state": model.state_dict(),
            "opt_state":   opt.state_dict(),
            "best_val_loss": best_val_loss,
            "history":     history,
            "extra_cfg":   extra_cfg,
        }, ckpt_path)
        print(f"  Checkpoint : {ckpt_path}")

    return history, n_params

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LARA Phase 2 -- v1 vs v2 Architecture (CPU)")
    print("="*60)
    print(f"  max_iters={CPU_TRAIN['max_iters']}  "
          f"block_size={CPU_TRAIN['block_size']}  "
          f"batch_size={CPU_TRAIN['batch_size']}")

    tc_obj = type("TC", (), CPU_TRAIN)()
    tc_obj.block_size = CPU_TRAIN["block_size"]
    tc_obj.batch_size = CPU_TRAIN["batch_size"]
    tc_obj.dataset    = CPU_TRAIN["dataset"]
    tc_obj.data_dir   = CPU_TRAIN["data_dir"]

    train_loader, val_loader = get_dataloaders(tc_obj)

    all_results = {}
    for seed_offset, (name, cls, extra) in enumerate(EXPERIMENTS):
        hist, n_params = run_experiment(name, cls, extra, train_loader, val_loader,
                                        seed=42 + seed_offset)
        all_results[name] = {"params": n_params, "final_ppl": hist[-1]["ppl"],
                             "init_ppl": hist[0]["ppl"], "best_ppl": min(h["ppl"] for h in hist)}

    print("\n" + "="*60)
    print("  RESULTATS FINAUX")
    print("="*60)
    print(f"  {'Modele':<25} {'Params':>10} {'PPL init':>10} {'best PPL':>10} {'Gain':>8}")
    print(f"  {'-'*67}")
    for name, r in all_results.items():
        gain = (r['init_ppl'] - r['best_ppl']) / r['init_ppl'] * 100
        print(f"  {name:<25} {r['params']:>10,} {r['init_ppl']:>10.1f} "
              f"{r['best_ppl']:>10.2f} {gain:>7.1f}%")

    print("\n  Resultats sauvegardes dans results/")
    print("  -> Phase 3 : lancer train_cpu_v2.py avec GPU pour benchmark officiel\n")
