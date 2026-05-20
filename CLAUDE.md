# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LARA (Latent Adaptive Reasoning Architecture) is a research framework implementing advanced transformer optimizations as independent, composable "briques" (building blocks). The goal is systematic ablation of each technique via controlled experiments.

## Commands

### Setup
```bash
pip install -r lara/requirements.txt
```

### Training
```bash
# Run a named experiment
python lara/train.py --experiment {baseline|diff_attn|mor|coconut|lara_full|lara_v2|lara_v2_full}

# With GPU and W&B logging
python lara/train.py --experiment lara_full --max_iters 10000 --wandb --device cuda

# CPU-only quick runs
python lara/train_cpu.py        # small model, ~15s/100 steps
python lara/train_cpu_v2.py     # advanced CPU variant
```

### Validation & Evaluation
```bash
python lara/smoke_test.py                      # quick sanity check (~30s on CPU)
python lara/evaluate.py                        # perplexity, throughput, KV compression
python lara/experiments/benchmark.py          # comparative benchmark across checkpoints
python lara/compare_results.py                # convergence analysis across experiments
```

### Running All Phase 1 Experiments
```bash
bash lara/experiments/run_all.sh
```

## Architecture

### Brique System

Each brique is independently toggled in `ModelConfig` (defined in [lara/config.py](lara/config.py)):

| Flag | Brique | Paper |
|------|--------|-------|
| `use_diff_attn` | Differential Attention (1a) | ICLR 2025, arxiv:2410.05258 |
| `use_mla` | Multi-Head Latent Attention (1b) | DeepSeek V3, arxiv:2412.19437 |
| `use_mor` | Mixture of Recursions (2) | NeurIPS 2025, arxiv:2507.10524 |
| `use_recurrent_depth` | Recurrent Depth Scaling (2b) | ICLR 2026, arxiv:2502.05171 |
| `use_coconut` | Coconut Latent Reasoning (3) | Meta FAIR, arxiv:2412.06769 |
| `use_titans` | Titans Neural Memory (4) | Google, arxiv:2501.00663 |

### Key Files

- [lara/model/lara.py](lara/model/lara.py) — main LARA model; `LARABlock` dynamically selects attention type and applies enabled briques
- [lara/model/baseline.py](lara/model/baseline.py) — standard GPT reference (nanoGPT-style)
- [lara/config.py](lara/config.py) — `ModelConfig` + `TrainConfig` + 7 preset experiment configs
- [lara/train.py](lara/train.py) — main training loop with gradient accumulation, bfloat16, `torch.compile()`, checkpoint saving
- [lara/data/dataset.py](lara/data/dataset.py) — dataset loading (Tiny Shakespeare / FineWeb-Edu 10B, tiktoken GPT2 encoding, uint16 binary files)

### Preset Experiment Configs (in config.py)

| Config function | Briques active |
|-----------------|---------------|
| `baseline_config()` | none |
| `diff_attn_config()` | 1a |
| `mor_config()` | 1a + 2 |
| `coconut_config()` | 1a + 2 + 3 |
| `lara_full_config()` | all (Phase 1) |
| `lara_v2_config()` | 1b + 2b + 4 |
| `lara_v2_full_config()` | lara_v2 + deferred Coconut fine-tune |

### Design Patterns

- **Weight sharing**: MoR recursion depths reuse the same transformer blocks for parameter efficiency.
- **Curriculum learning**: Coconut gradually injects latent thinking via a sigmoid gate controlled by training step.
- **Auxiliary losses**: Router load-balancing loss (MoR) and memory update loss (Titans) are summed into the main loss.
- **Test-time scaling**: `use_recurrent_depth` enables running more recursions at inference than at training depth.
- **Attention dispatch**: `LARABlock` selects standard → DiffAttn → MLA based on config flags; only one attention variant is active at a time.

### Data

- Datasets are tokenized once to binary `uint16` files with a 90/10 train/val split.
- Tiny Shakespeare (~1M tokens) is used for fast iteration; FineWeb-Edu 10B for full runs.
- Checkpoints and results are saved to `lara/results/`.
