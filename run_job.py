"""
Coller la commande ci-dessous et cliquer Run dans Lightning Studio.
L'exécution continue même si le studio se ferme.
"""
import subprocess, sys, os

# ── COLLER LA COMMANDE ICI ────────────────────────────────────────────────────
CMD = """
train.py \
  --experiment lara_v2_dca \
  --dataset fineweb \
  --n_embd 1024 --n_layer 6 --n_head 8 --block_size 512 \
  --n_recursions 4 \
  --learning_rate 3e-4 --warmup_iters 500 \
  --batch_size 8 --grad_accum 16 \
  --max_iters 2000 --compile --device cuda
"""
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
args = CMD.replace("\\\n", " ").split()
args[0] = os.path.join(ROOT, args[0])

subprocess.run([sys.executable] + args, check=True, cwd=ROOT)
