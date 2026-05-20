"""
Coconut Fine-Tune (Phase 2).

Strategy: take a converged LARA checkpoint (Phase 1, no Coconut),
then gradually enable Coconut latent reasoning with a slow curriculum
and reduced learning rate.

This avoids the gradient shock seen when Coconut activates mid-training
because the base model has already converged to a good solution.

Usage:
    python train_coconut_ft.py --checkpoint results/B_--_LARA_v1-3_ckpt.pt
    python train_coconut_ft.py --checkpoint results/D_--_LARA_v2_v2_ckpt.pt
"""
import sys, os, math, time, json, argparse
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from config import ModelConfig
from model.lara import LARA
from data.dataset import get_dataloaders


# ── Fine-tune config ──────────────────────────────────────────
FT_TRAIN = dict(
    dataset    = "local",
    data_dir   = "data",
    batch_size = 6,
    block_size = 128,
    max_iters  = 1000,
    learning_rate = 3e-5,    # 10x lower than Phase 1
    weight_decay  = 0.1,
    grad_clip     = 1.0,
    warmup_iters  = 50,      # short warmup
    lr_decay_iters = 1000,
    min_lr     = 3e-6,
    eval_interval = 100,
    eval_iters    = 50,
    log_interval  = 50,
)

# Coconut curriculum for fine-tune: gate starts near-zero, activates late
# thought_gate init = sigmoid(-4.6)≈0.01 so first steps are nearly no-ops
FT_COCONUT = dict(
    n_thinking_steps   = 4,
    coconut_ramp_steps = 800,    # ramp over last 800 iters
    coconut_start_iter = 200,    # first thinking step appears at iter 200+800/4=400
)


def get_lr(it, tc):
    if it < tc["warmup_iters"]:
        return tc["learning_rate"] * it / tc["warmup_iters"]
    if it > tc["lr_decay_iters"]:
        return tc["min_lr"]
    progress = (it - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
    return tc["min_lr"] + 0.5 * (1 + math.cos(math.pi * progress)) * (tc["learning_rate"] - tc["min_lr"])


@torch.no_grad()
def evaluate(model, loader, n=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        out = model(x, y)
        losses.append(out[1].item())
    return sum(losses) / len(losses)


def load_and_enable_coconut(ckpt_path):
    """Load checkpoint and enable Coconut fine-tuning."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra_cfg = ckpt.get("extra_cfg", {})
    base_cfg = dict(
        n_embd=256, n_head=4, n_layer=6, block_size=128,
        dropout=0.0, bias=False, memory_size=128,
        n_recursions=3, mor_top_k=0.5, memory_lr=0.01,
    )
    # Enable Coconut with slow curriculum for fine-tune
    full_cfg = {**base_cfg, **extra_cfg, **FT_COCONUT, "use_coconut": True}
    mc = ModelConfig(**full_cfg)

    print(f"  Coconut first activation at iter: {mc.coconut_ramp_steps // mc.n_thinking_steps}")

    model = LARA(mc)
    # Load pre-trained weights (non-strict: Coconut params are new)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Loaded weights | new params: {missing} | unexpected: {unexpected}")

    prev_ppl = math.exp(ckpt.get("best_val_loss", 99))
    return model, ckpt.get("history", []), prev_ppl


def run_finetune(ckpt_path, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  Coconut Fine-Tune Phase 2")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"{'='*60}")

    model, base_history, prev_ppl = load_and_enable_coconut(ckpt_path)
    print(f"  Base PPL (Phase 1): {prev_ppl:.2f}")
    print(f"  Params: {model.num_params():,}")

    tc = FT_TRAIN
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    train_iter = iter(train_loader)
    history = []
    best_val_loss = float("inf")
    best_ppl = float("inf")
    t_start = time.time()

    for it in range(tc["max_iters"] + 1):
        lr = get_lr(it, tc)
        for pg in opt.param_groups: pg["lr"] = lr

        if it % tc["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, tc["eval_iters"])
            model.train()
            ppl = math.exp(val_loss)
            elapsed = time.time() - t_start
            n_think = model._coconut_curriculum.step(it)
            flag = " *" if val_loss < best_val_loss else ""
            print(f"  [{it:5d}/{tc['max_iters']}]  val_loss={val_loss:.4f}  "
                  f"ppl={ppl:.2f}  think={n_think}  lr={lr:.1e}  {elapsed:.0f}s{flag}")
            history.append({"iter": it, "val_loss": val_loss, "ppl": ppl, "n_think": n_think})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ppl = ppl

        if it == tc["max_iters"]: break

        try:
            x, y = next(train_iter)
        except StopIteration:
            if hasattr(model, "memory") and model.memory is not None:
                model.memory.reset_memory()
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        opt.zero_grad(set_to_none=True)
        out = model(x, y, iter_num=it)
        loss = out[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()

        if it % tc["log_interval"] == 0 and it > 0:
            n_think = model._coconut_curriculum.step(it)
            print(f"  [{it:5d}]  train_loss={loss.item():.4f}  think={n_think}")

    print(f"  Meilleur val PPL (Phase 2): {best_ppl:.2f}  (base: {prev_ppl:.2f})")

    base_name = os.path.basename(ckpt_path).replace("_ckpt.pt", "").replace("_v2_ckpt.pt", "")
    out_name = f"results/{base_name}_coconut_ft"
    os.makedirs("results", exist_ok=True)
    with open(f"{out_name}.json", "w") as f:
        json.dump({"history_base": base_history, "history_ft": history,
                   "best_ppl_ft": best_ppl, "best_ppl_base": prev_ppl}, f, indent=2)

    # Persist extra_cfg so evaluate.py can reconstruct the model correctly
    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ft_extra_cfg = {**base_ckpt.get("extra_cfg", {}), **FT_COCONUT, "use_coconut": True}
    torch.save({
        "model_state":   model.state_dict(),
        "opt_state":     opt.state_dict(),
        "best_val_loss": best_val_loss,
        "history":       history,
        "extra_cfg":     ft_extra_cfg,
    }, f"{out_name}_ckpt.pt")
    print(f"  Saved: {out_name}_ckpt.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to Phase 1 checkpoint .pt file")
    args = p.parse_args()

    tc_obj = type("TC", (), {**FT_TRAIN, "dataset": "local", "data_dir": "data"})()
    tc_obj.block_size = FT_TRAIN["block_size"]
    tc_obj.batch_size = FT_TRAIN["batch_size"]
    tc_obj.dataset    = "local"
    tc_obj.data_dir   = "data"

    train_loader, val_loader = get_dataloaders(tc_obj)
    run_finetune(args.checkpoint, train_loader, val_loader)
