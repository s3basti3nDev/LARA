# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LARA (Latent Adaptive Reasoning Architecture) is a research framework implementing advanced transformer optimizations as independent, composable "briques" (building blocks). The goal is systematic ablation of each technique via controlled experiments and publication of a scientific paper.

**GitHub**: https://github.com/s3basti3nDev/LARA  
**Training**: Lightning.ai (NVIDIA L4, 23GB VRAM) — code at `/teamspace/studios/this_studio/`  
**Evaluation**: `python evaluate.py --model checkpoints/<name>_best.pt --dataset fineweb`

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
# Run a named experiment (from repo root)
python train.py --experiment {baseline|diff_attn|mor|coconut|lara_full|lara_v2|lara_v2_dca|lara_v2_rope}

# Standard full run (Lightning.ai)
python train.py --experiment diff_attn \
  --dataset fineweb \
  --n_embd 1024 --n_layer 6 --n_head 8 --block_size 512 \
  --learning_rate 3e-4 --warmup_iters 500 \
  --batch_size 8 --grad_accum 16 \
  --max_iters 5000 --compile --device cuda

# Resume from interrupted run (use _last.pt, not _best.pt)
python train.py --experiment diff_attn ... --resume checkpoints/exp_b_diff_attn_last.pt

# Via Lightning Run button
python run_job.py   # paste command inside CMD = """ ... """
```

### Evaluation & Benchmarks
```bash
python smoke_test.py                           # quick sanity check (~30s CPU)
python evaluate.py --model checkpoints/X_best.pt --dataset fineweb
python run_benchmarks.py --lara checkpoints/X_best.pt --device cuda
```

## Architecture

### Brique System

Each brique is independently toggled in `ModelConfig` (defined in [config.py](config.py)):

| Flag | Brique | Paper |
|------|--------|-------|
| `use_diff_attention` | Differential Attention (1a) | ICLR 2025, arxiv:2410.05258 |
| `use_mla` | Multi-Head Latent Attention (1b) | DeepSeek V3, arxiv:2412.19437 |
| `use_mor` | Mixture of Recursions (2) | NeurIPS 2025, arxiv:2507.10524 |
| `use_recurrent_depth` | Recurrent Depth Scaling (2b) | ICLR 2026, arxiv:2502.05171 |
| `use_coconut` | Coconut Latent Reasoning (3) | Meta FAIR, arxiv:2412.06769 |
| `use_titans` | Titans Neural Memory (4) | Google, arxiv:2501.00663 |
| `use_dca` | Depth Cross-Attention (2c) | Original — inspiré DREAMER arxiv:2601.21582 |
| `use_rope` | Rotary Position Embeddings (5) | Su et al. 2021, arxiv:2104.09864 |

### Key Files (repo root, flat structure since 2026-05-21)

- [model/lara.py](model/lara.py) — main LARA model; `LARABlock` dynamically selects attention type
- [model/baseline.py](model/baseline.py) — standard GPT reference (nanoGPT-style)
- [model/rope.py](model/rope.py) — `RotaryEmbedding` module, used by MLA and DiffAttn
- [model/depth_attention.py](model/depth_attention.py) — `DepthCrossAttention` (DCA brique)
- [config.py](config.py) — `ModelConfig` + `TrainConfig` + 9 preset experiment configs
- [train.py](train.py) — training loop: grad accum, bfloat16, torch.compile, `--resume`
- [data/dataset.py](data/dataset.py) — FineWeb-Edu streaming, 500M token cap for Colab
- [run_job.py](run_job.py) — Lightning.ai Run button script (paste command in CMD)

### Preset Experiment Configs

| Config function | Experiment | Briques active |
|-----------------|------------|---------------|
| `baseline_config()` | exp_a | none |
| `diff_attn_config()` | exp_b | 1a |
| `mor_config()` | exp_c | 1a + 2 |
| `coconut_config()` | exp_d | 1a + 2 + 3 |
| `lara_full_config()` | exp_e | all Phase 1 |
| `lara_v2_config()` | exp_f | 1b + 2b + 4 |
| `lara_v2_full_config()` | exp_g | lara_v2 + Coconut deferred |
| `lara_v2_dca_config()` | exp_h | 1b + 2b + 4 + DCA |
| `lara_v2_rope_config()` | exp_i | 1b + 2b + 4 + DCA + RoPE |

### Design Patterns

- **Weight sharing**: MoR/RecurrentDepth reuse same transformer blocks — enables test-time depth scaling.
- **RoPE**: replaces absolute `wpe` embedding; applied to Q/K inside MLA and DiffAttn when `use_rope=True`.
- **DCA gate**: starts at sigmoid(-4.6) ≈ 0.01, ramps up during training — no-op initially.
- **Curriculum learning**: Coconut gradually injects latent thinking via sigmoid gate.
- **Auxiliary losses**: MoR router load-balancing + Titans memory loss summed into main loss.
- **Checkpoints**: `_best.pt` = best val_loss ever; `_last.pt` = most recent eval (use for resume).
- **Resume**: `--resume checkpoints/exp_X_last.pt` restores model + optimizer + iter_num.

### Data

- FineWeb-Edu 10B streamed and tokenized to uint16 binary files.
- Cap: 500M train tokens by default (≈3 min on Colab T4). Pass `max_train_tokens=0` for full dataset.
- Fallback: detects old-format `fineweb_sample-10BT_train.bin` (Lightning.ai persistent disk).
- Checkpoints saved to `checkpoints/`.

## Ablation Results (mai 2026) — COMPLETS

PPL mesuré avec evaluate.py (block_size=512, fidèle à l'entraînement).

| # | Experiment | Corpus | Briques | Params | Val PPL | Tok/s | KV cache | Iters | Statut |
|---|-----------|--------|---------|--------|---------|-------|----------|-------|--------|
| A | baseline | FineWeb | — | 203M | ~52* | 700 | 1x | 5000 | ✅ |
| B | diff_attn | FineWeb | DiffAttn | 127.5M | 70.56 | 1,231 | 1x | 5000 | ✅ |
| C | mor | FineWeb | DiffAttn+MoR | 127.5M | 74.52 | 433 | 1x | 5000 | ✅ |
| D | coconut | FineWeb | DiffAttn+MoR+Coconut | 128.5M | ~90* | 357 | 1x | 5000 | ✅ |
| E | lara_full | FineWeb | Phase 1 complète | 135.9M | ~80* | 334 | 1x | 5000 | ✅ |
| F | lara_v2 | FineWeb | MLA+RD+Titans | 190M | ~58* | 395 | 16x | 5000 | ✅ |
| G | lara_v2_full | — | +Coconut | — | — | — | — | — | ⏭ skipped |
| H | lara_v2_dca | FineWeb | +DCA | 125.1M | 60.84 | 486 | 16x | 5000 | ✅ |
| I | lara_v2_rope (5k) | FineWeb | +DCA+RoPE | 124.6M | ~52* | 472 | 16x | 5000 | ✅ |
| I | lara_v2_rope (50k FW) | FineWeb | +DCA+RoPE | 124.6M | 14.66 | 472 | 16x | 50000 | ✅ |
| I | lara_v2_rope (50k C4) | C4 | +DCA+RoPE | 124.6M | **13.14** | 440 | 16x | 50000 | ✅ |

*checkpoints perdus — PPL estimé depuis val_loss enregistré durant l'entraînement

**Résultat clé** : lara_v2_rope (50k C4) atteint PPL **13.14** avec 124.6M params, 16x compression KV, 500M tokens sur corpus C4 diversifié.

**Bug corrigé** : evaluate.py utilisait block_size=128 au lieu du block_size du modèle (512) — tous les anciens PPL étaient surestimés. Valeurs corrigées disponibles pour B, C, H, I.

**Observations Phase 1** : DiffAttn seul domine (70.56). MoR dégrade légèrement (74.52). Coconut nécessite >5k iters. lara_full ne bénéficie pas de synergie à iso-iters.

**Note evaluate.py** : le fix (block_size=mc.block_size) est dans evaluate.py depuis le 27 mai 2026.

## Benchmarks lm_eval (0-shot)

| Model | Params | Corpus | Iters | hellaswag | arc_easy | lambada |
|-------|--------|--------|-------|-----------|----------|---------|
| pythia-160m | 160M | 300B tokens | — | 30.18% | 39.81% | 32.89% |
| gpt2 | 117M | 40B tokens | — | 31.08% | 39.60% | 32.10% |
| diff_attn | 127.5M | FineWeb 500M | 5k | 26.47% | 35.27% | 8.21% |
| mor | 127.5M | FineWeb 500M | 5k | 26.62% | 34.89% | 7.67% |
| lara_v2_dca | 125.1M | FineWeb 500M | 5k | 26.30% | 36.78% | 9.33% |
| lara_v2_rope | 124.6M | FineWeb 500M | 5k | 26.45% | 34.55% | 7.63% |
| lara_v2_rope | 124.6M | FineWeb 500M | 50k | 25.45% | 29.67% | 1.59% |
| lara_v2_rope | 124.6M | C4 500M | 50k | 26.34% | 26.73% | **4.50%** |

**Observation corpus** : PPL et benchmarks divergent après sur-entraînement (50k iters sur 500M tokens = ~30 passes). FineWeb-Edu aide arc_easy (contenu éducatif corrélé aux questions de science) ; C4 aide lambada (texte web naturel). La solution pour les benchmarks est plus de tokens uniques, pas plus d'iters.

## Infrastructure

- **torch.compile()** : 2-5 min silencieuses avant le premier iter — normal.
- **torch.compile() + bfloat16** : T4 Colab ne supporte pas bfloat16 natif → très lent (25s/iter). Utiliser Lightning L4.
- **evaluate.py** : gère le préfixe `_orig_mod.` introduit par torch.compile().
- **Lightning Run button** : le job survit à la mise en veille du studio via `run_job.py`.
- **Git workflow Lightning** : `git pull origin main` depuis `/teamspace/studios/this_studio/`.
