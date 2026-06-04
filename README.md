# LARA — Latent Adaptive Reasoning Architecture

A research framework for the **systematic, controlled ablation of modern transformer
improvements**, implemented as independent, composable modules (*briques*) that can be
toggled on or off from a single shared configuration.

LARA is not a competitive small language model. It is an **experimental harness**: every
improvement is a flag in one `ModelConfig`, and all configurations are trained under
identical conditions (dataset, optimizer, batch size, schedule, parameter and token
budgets) so that measured differences can be attributed to architectural choices rather
than to training artifacts.

---

## What this repo answers

Given a set of published transformer improvements, **which subset should you adopt, and do
their benefits actually compose?** LARA implements eight techniques as toggles and runs
controlled ablations to find out — on a single GPU, at small scale.

### Two findings

1. **Composition is not automatic.** At short training horizons, several combinations
   provide no additive benefit; only a specific subset (MLA + Depth Cross-Attention + RoPE)
   composes without interference.
2. **Perplexity and downstream accuracy can diverge.** Under a fixed token budget traversed
   many times (~30 epochs over 500M tokens), continued training keeps lowering held-out
   perplexity *on the training distribution* while zero-shot benchmark accuracy stagnates or
   degrades. We attribute this to distribution overfitting under repeated data, and find
   that **corpus composition is a stronger lever on benchmark accuracy than training
   duration.**

> **Read perplexity with care.** The 13.14 PPL reported below is measured on a C4 held-out
> split — *in-distribution* with respect to training. It is a within-study reference point
> across our own configurations, **not** a cross-model claim. Models trained on other
> corpora (e.g. Pythia-160M on the Pile) are not directly comparable on this metric; on
> downstream benchmarks they remain ahead (see the benchmark table).

---

## The eight composable briques

| Brique | Technique | Role |
|---|---|---|
| DiffAttn | Differential Attention | Attention (noise suppression) |
| MLA | Multi-Head Latent Attention | Attention + 16× KV-cache compression |
| MoR | Mixture of Recursions | Adaptive depth (token routing) |
| RecDepth | Recurrent Depth Scaling | Test-time depth scaling via weight sharing |
| Coconut | Coconut latent reasoning | Continuous latent "thought" vectors |
| Titans | Titans neural memory | Long-range associative memory |
| RoPE | Rotary Position Embeddings | Positional encoding |
| DCA | Depth Cross-Attention | Cross-attention to a per-layer depth embedding (our variant) |

MLA, MoR, RecDepth, Coconut, Titans, RoPE and Differential Attention are published
techniques that we re-implement and integrate. **DCA is the only brique introduced here**,
and is presented as an exploratory variant rather than a validated contribution.

---

## Results

### Ablations at 5,000 iterations (~330M tokens, FineWeb-Edu)

| # | Config | Briques | Params | PPL ↓ | Tok/s ↑ | KV | Iters |
|---|---|---|---|---|---|---|---|
| A | baseline | — | 203M | ~52* | 700 | 1× | 5k |
| B | diff_attn | DiffAttn | 127.5M | 70.56 | 1,231 | 1× | 5k |
| C | mor | DiffAttn+MoR | 127.5M | 74.52 | 433 | 1× | 5k |
| D | coconut | +Coconut | 128.5M | ~90* | 357 | 1× | 5k |
| E | lara_full | Phase-1 complete | 135.9M | ~80* | 334 | 1× | 5k |
| F | lara_v2 | MLA+RecDepth+Titans | 190M | ~58* | 395 | 16× | 5k |
| H | lara_v2_dca | +DCA | 125.1M | 60.84 | 486 | 16× | 5k |
| I | lara_v2_rope | +RoPE | 124.6M | ~52* | 472 | 16× | 5k |

`*` = checkpoint lost, PPL estimated from logged val_loss.

### Extended training (50,000 iterations, best config `lara_v2_rope`)

| Corpus | Params | PPL ↓ | KV |
|---|---|---|---|
| FineWeb-Edu | 124.6M | 14.66 | 16× |
| **C4** | **124.6M** | **13.14** | **16×** |

### 0-shot downstream benchmarks

| Model | Corpus (Iters) | Params | HellaSwag | ARC-E | LAMBADA |
|---|---|---|---|---|---|
| Pythia-160M † | 300B tokens | 160M | 30.18% | 39.81% | 32.89% |
| GPT-2 † | 40B tokens | 117M | 31.08% | 39.60% | 32.10% |
| diff_attn | FW-Edu (5k) | 127.5M | 26.47% | 35.27% | 8.21% |
| mor | FW-Edu (5k) | 127.5M | 26.62% | 34.89% | 7.67% |
| lara_v2_dca | FW-Edu (5k) | 125.1M | 26.30% | 36.78% | 9.33% |
| lara_v2_rope | FW-Edu (5k) | 124.6M | 26.45% | 34.55% | 7.63% |
| lara_v2_rope | FW-Edu (50k) | 124.6M | 25.45% | 29.67% | 1.59% |
| lara_v2_rope | C4 (50k) | 124.6M | 26.34% | 26.73% | 4.50% |

`†` = trained on much larger token budgets. These models are the honest baseline:
our small, data-constrained models do not match them on downstream tasks, which is
precisely the point of the perplexity–benchmark divergence analysis.

---

## Setup

All experiments ran on a **single NVIDIA L4 GPU (23 GB VRAM)**. Models use `d = 1024`,
`n_layer = 6`, `n_head = 8`, block size 512, GPT-2 BPE tokenizer, AdamW with a cosine
schedule. Short runs use FineWeb-Edu; long runs compare FineWeb-Edu and C4.

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA 12.x (for GPU runs).

```bash
git clone https://github.com/s3basti3nDev/LARA.git
cd LARA
pip install -r requirements.txt
python smoke_test.py   # quick sanity check (~30s, CPU)
```

## Reproducing the experiments

Each configuration (A–I) is a preset in `config.py`. All presets share the same
`train.py` entry point; the `--experiment` flag selects the brique combination.

```bash
# Standard run (Lightning.ai / single GPU)
python train.py --experiment lara_v2_rope \
  --dataset fineweb \
  --n_embd 1024 --n_layer 6 --n_head 8 --block_size 512 \
  --learning_rate 3e-4 --warmup_iters 500 \
  --batch_size 8 --grad_accum 16 \
  --max_iters 5000 --compile --device cuda

# Available experiments: baseline | diff_attn | mor | coconut |
#                        lara_full | lara_v2 | lara_v2_dca | lara_v2_rope

# Evaluate perplexity
python evaluate.py --model checkpoints/exp_i_lara_v2_rope_best.pt --dataset fineweb

# Run 0-shot benchmarks (HellaSwag, ARC-Easy, LAMBADA)
python run_benchmarks.py --lara checkpoints/exp_i_lara_v2_rope_best.pt --device cuda
```

### ModelConfig flags

All briques are controlled by boolean flags in `ModelConfig` (`config.py`):

| Flag | Default | Brique |
|------|---------|--------|
| `use_diff_attention` | `False` | DiffAttn |
| `use_mla` | `False` | Multi-Head Latent Attention (16× KV) |
| `mla_kv_compress` | `0` (auto) | MLA compression dim (0 = n_embd // 16) |
| `use_mor` | `False` | Mixture of Recursions |
| `use_recurrent_depth` | `False` | Recurrent Depth Scaling |
| `n_recursions` | `4` | Max recursion depth R |
| `use_coconut` | `False` | Coconut latent reasoning |
| `n_thinking_steps` | `4` | Coconut thought passes |
| `use_titans` | `False` | Titans neural memory |
| `memory_size` | `512` | Titans memory slots |
| `use_dca` | `False` | Depth Cross-Attention (ours) |
| `dca_n_head` | `4` | DCA attention heads |
| `use_rope` | `False` | Rotary Position Embeddings |

## Repository structure

```
.
├── model/
│   ├── baseline.py         # GPT reference (nanoGPT-style)
│   ├── lara.py             # main LARA model + LARABlock
│   ├── diff_attention.py   # Brique 1a — Differential Attention
│   ├── mla.py              # Brique 1b — Multi-Head Latent Attention
│   ├── recursion.py        # Brique 2/2b — MoR + Recurrent Depth
│   ├── depth_attention.py  # Brique 2c — Depth Cross-Attention (ours)
│   ├── coconut.py          # Brique 3 — Coconut curriculum
│   ├── memory.py           # Brique 4 — Titans neural memory
│   └── rope.py             # Brique 5 — Rotary Position Embeddings
├── data/
│   └── dataset.py          # FineWeb-Edu / C4 streaming dataloader
├── config.py               # ModelConfig + TrainConfig + 9 preset configs
├── train.py                # training loop (grad accum, bfloat16, compile)
├── evaluate.py             # perplexity evaluation
├── run_benchmarks.py       # lm-eval harness runner
├── lm_eval_wrapper.py      # lm-eval adapter for LARA
├── smoke_test.py           # quick sanity check
├── run_job.py              # Lightning.ai Run button launcher
└── requirements.txt
```

---

## Citation

```bibtex
@article{tamagno2026lara,
  title   = {LARA: Composable Transformer Improvements via Systematic Ablation},
  author  = {Tamagno, S{\'e}bastien},
  year    = {2026},
  journal = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

## Acknowledgments

Developed at TMG Consulting. This project used Claude (claude.ai/code, Anthropic) as a
coding and writing assistant; all scientific decisions, experimental design, and
conclusions are the author's own.

## License

MIT License — see [LICENSE](LICENSE) for details.
