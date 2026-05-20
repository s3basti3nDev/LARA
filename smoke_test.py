"""Quick smoke test + mini training for all LARA variants — CPU-friendly."""
import sys, time, torch
sys.path.insert(0, ".")
from config import baseline_config, diff_attn_config, mor_config, lara_full_config
from model.baseline import GPT
from model.lara import LARA

torch.manual_seed(42)

# Petit modèle pour test rapide sur CPU
VOCAB, T, B = 64, 32, 4
N_STEPS = 50   # suffit pour vérifier que la loss descend

EXPERIMENTS = [
    ("A  Baseline GPT         ", baseline_config,  GPT),
    ("B  + Diff Attention     ", diff_attn_config,  LARA),
    ("C  + Mixture Recursions ", mor_config,        LARA),
    ("E  LARA Full            ", lara_full_config,  LARA),
]

def small_cfg(cfg_fn):
    mc, _ = cfg_fn()
    mc.vocab_size  = VOCAB
    mc.block_size  = T
    mc.n_embd      = 128   # réduit pour CPU
    mc.n_head      = 4
    mc.n_layer     = 4
    mc.memory_size = 64    # Titans plus léger
    return mc

# ── Forward pass ─────────────────────────────────────────────
print("\n=== [1/2] Forward pass ===")
x = torch.randint(0, VOCAB, (B, T))
for name, cfg_fn, cls in EXPERIMENTS:
    mc = small_cfg(cfg_fn)
    m = cls(mc)
    t0 = time.time()
    out = m(x, x) if cls == GPT else m(x, x)
    ms = (time.time() - t0) * 1000
    loss = out[1]
    print(f"  {name}  params={m.num_params():>7,}  loss={loss.item():.4f}  {ms:.0f}ms  OK")

# ── Mini-training ─────────────────────────────────────────────
print(f"\n=== [2/2] Mini-training ({N_STEPS} steps) ===")
data = torch.randint(0, VOCAB, (B * 100, T))

for name, cfg_fn, cls in EXPERIMENTS:
    mc = small_cfg(cfg_fn)
    m = cls(mc)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    t0 = time.time()
    for i in range(N_STEPS):
        x = data[torch.randint(0, len(data), (B,))]
        opt.zero_grad(set_to_none=True)
        out = m(x, x) if cls == GPT else m(x, x, iter_num=i)
        loss = out[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    elapsed = time.time() - t0
    ok = "OK" if last < first else "WARN"
    print(f"  {name}  init={first:.3f}  final={last:.3f}  "
          f"delta={first-last:+.3f}  {elapsed:.1f}s  {ok}")

print("\nSmoke test termine.\n")
