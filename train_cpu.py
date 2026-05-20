"""
Entraînement CPU-friendly pour comparer baseline vs LARA Full.
Config petite mais representativa — les courbes de convergence extrapolent
à grande échelle via les lois de scaling (Chinchilla).

Résultats attendus : perplexité finale + courbe de convergence comparative.
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
# n_embd=256, block_size=128 → ~15s/100steps sur CPU basique

CPU_MODEL = dict(
    n_embd     = 256,
    n_head     = 4,
    n_layer    = 6,
    block_size = 128,
    dropout    = 0.0,
    bias       = False,
    memory_size = 128,
    n_recursions = 3,
    n_thinking_steps = 3,
    mor_top_k  = 0.5,
    memory_lr  = 0.01,
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
    ("A — Baseline GPT    ", GPT,  dict()),
    ("B — LARA-3 (no Coco)", LARA, dict(
        use_diff_attention = True,
        use_mor            = True,
        use_coconut        = False,   # Coconut nécessite pre-training séparé
        use_titans         = True,
    )),
    ("C — LARA-4 (all)    ", LARA, dict(
        use_diff_attention = True,
        use_mor            = True,
        use_coconut        = True,
        use_titans         = True,
        n_thinking_steps   = 2,       # ramp activera step ~1334, hors de notre range
    )),
]

# ── Helpers ──────────────────────────────────────────────────
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
        out = model(x, y) if not is_lara else model(x, y)
        losses.append(out[1].item())
    model.eval()
    return sum(losses) / len(losses)

def make_model(cls, extra_cfg):
    mc = ModelConfig(**{**CPU_MODEL, **extra_cfg})
    return cls(mc)

def run_experiment(name, cls, extra_cfg, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    is_lara = cls == LARA
    model = make_model(cls, extra_cfg)
    n_params = model.num_params()
    print(f"  Paramètres : {n_params:,}")

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
        # LR schedule
        lr = get_lr(it, tc)
        for pg in opt.param_groups: pg["lr"] = lr

        # Eval
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

        # Batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            # Fix 3 : reset Titans memory at each epoch boundary
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
    safe_name = name.replace(" ", "_").replace("—", "-").strip()
    with open(f"results/{safe_name}.json", "w") as f:
        json.dump({"name": name, "params": n_params, "history": history}, f, indent=2)

    # Save checkpoint for potential Coconut fine-tune Phase 2
    if is_lara:
        ckpt_path = f"results/{safe_name}_ckpt.pt"
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
    print("  LARA Phase 1 — Benchmark Shakespeare (CPU)")
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
    for name, cls, extra in EXPERIMENTS:
        hist, n_params = run_experiment(name, cls, extra, train_loader, val_loader)
        all_results[name] = {"params": n_params, "final_ppl": hist[-1]["ppl"],
                             "init_ppl": hist[0]["ppl"]}

    print("\n" + "="*60)
    print("  RÉSULTATS FINAUX")
    print("="*60)
    print(f"  {'Modèle':<25} {'Params':>10} {'PPL init':>10} {'PPL final':>10} {'Gain':>8}")
    print(f"  {'-'*65}")
    ppls = list(all_results.values())
    for name, r in all_results.items():
        gain = (r['init_ppl'] - r['final_ppl']) / r['init_ppl'] * 100
        print(f"  {name:<25} {r['params']:>10,} {r['init_ppl']:>10.1f} "
              f"{r['final_ppl']:>10.2f} {gain:>7.1f}%")

    print("\n  Resultats sauvegardes dans results/")
    print("  -> Extrapolation GPU : diviser le temps par ~100-200x (A100)")
    print("  -> Extrapolation scale : lois de Chinchilla applicables\n")
