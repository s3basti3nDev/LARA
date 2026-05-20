from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 512
    n_head: int = 8           # must be even for DiffAttn
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False

    # --- Brique 1a : Differential Attention (ICLR 2025) ---
    use_diff_attention: bool = False

    # --- Brique 1b : Multi-Head Latent Attention (DeepSeek V3, 2024) ---
    # Replaces DiffAttn when enabled — ~16× smaller KV-cache.
    # mla_kv_compress=0 → auto = n_embd // 8
    use_mla: bool = False
    mla_kv_compress: int = 0

    # --- Brique 2 : Mixture of Recursions (NeurIPS 2025) ---
    use_mor: bool = False
    n_recursions: int = 4     # max recursion depth R
    mor_top_k: float = 0.5    # fraction of tokens routed to depth > 1
    # Recurrent Depth mode (arxiv:2502.05171): all tokens get R iterations,
    # no router — enables test-time depth scaling beyond the trained R.
    use_recurrent_depth: bool = False

    # --- Brique 3 : Coconut - Continuous Latent Reasoning (Meta, 2024) ---
    use_coconut: bool = False
    n_thinking_steps: int = 4   # latent "thought" passes before decoding
    coconut_ramp_steps: int = 2000
    # Start Coconut curriculum only after this many iters (phase 1 = 0 thinking)
    # Set coconut_start_iter > max_iters to defer Coconut to a fine-tune phase.
    coconut_start_iter: int = 0

    # --- Brique 4 : Titans - Test-Time Memory (Google, 2025) ---
    use_titans: bool = False
    memory_size: int = 512    # M slots in external memory
    memory_lr: float = 0.01   # inner-loop LR for memory update

    # --- Brique 2c : Depth Cross-Attention (inspiré DREAMER, arxiv:2601.21582) ---
    # Permet à chaque récursion d'assister les résumés des profondeurs précédentes.
    # Requiert use_mor=True ou use_recurrent_depth=True.
    use_dca: bool = False
    dca_compress: int = 0   # 0 = auto = n_embd // 8
    dca_n_head: int = 4

    # --- Brique 5 : BitNet 1.58 (Microsoft, 2025) ---
    use_bitnet: bool = False


@dataclass
class TrainConfig:
    # Data
    dataset: str = "shakespeare"   # "shakespeare" | "fineweb"
    data_dir: str = "data"

    # Batching
    batch_size: int = 12
    block_size: int = 1024
    gradient_accumulation_steps: int = 1

    # Optimisation
    max_iters: int = 5000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule (cosine with warmup)
    warmup_iters: int = 200
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5

    # Eval
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50

    # Logging
    wandb_log: bool = False
    wandb_project: str = "lara"
    run_name: str = "baseline"

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False


# ---- Preset configs for each experiment ----

def baseline_config() -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig()
    t = TrainConfig(run_name="exp_a_baseline")
    return m, t


def diff_attn_config() -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(use_diff_attention=True)
    t = TrainConfig(run_name="exp_b_diff_attn")
    return m, t


def mor_config() -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(use_diff_attention=True, use_mor=True, n_layer=6, n_recursions=4)
    t = TrainConfig(run_name="exp_c_mor")
    return m, t


def coconut_config() -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(use_diff_attention=True, use_mor=True, use_coconut=True,
                    n_layer=6, n_recursions=4, n_thinking_steps=4)
    t = TrainConfig(run_name="exp_d_coconut")
    return m, t


def lara_full_config() -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(
        use_diff_attention=True,
        use_mor=True,
        use_coconut=True,
        use_titans=True,
        n_layer=6,
        n_recursions=4,
        n_thinking_steps=4,
        memory_size=512,
    )
    t = TrainConfig(run_name="exp_e_lara_full")
    return m, t


def lara_v2_config() -> tuple[ModelConfig, TrainConfig]:
    """LARA v2: MLA (16× KV cache) + RecurrentDepth + Titans."""
    m = ModelConfig(
        use_mla=True,
        use_recurrent_depth=True,
        use_titans=True,
        n_layer=6,
        n_recursions=4,
        memory_size=512,
    )
    t = TrainConfig(run_name="exp_f_lara_v2")
    return m, t


def lara_v2_dca_config() -> tuple[ModelConfig, TrainConfig]:
    """LARA v2 + DCA (Depth Cross-Attention) — Brique 2c."""
    m = ModelConfig(
        use_mla=True,
        use_recurrent_depth=True,
        use_titans=True,
        use_dca=True,
        n_layer=6,
        n_recursions=4,
        memory_size=512,
    )
    t = TrainConfig(run_name="exp_h_lara_v2_dca")
    return m, t


def lara_v2_full_config() -> tuple[ModelConfig, TrainConfig]:
    """LARA v2 full: MLA + RecurrentDepth + Coconut (deferred) + Titans."""
    m = ModelConfig(
        use_mla=True,
        use_recurrent_depth=True,
        use_coconut=True,
        use_titans=True,
        n_layer=6,
        n_recursions=4,
        n_thinking_steps=4,
        coconut_ramp_steps=3000,
        coconut_start_iter=2000,  # defer to fine-tune phase
        memory_size=512,
    )
    t = TrainConfig(run_name="exp_g_lara_v2_full")
    return m, t
