"""
Differential Attention — ICLR 2025 (Microsoft / Tsinghua).
arxiv: 2410.05258

Each differential head uses two attention maps (A1 - λ*A2).
The subtraction cancels common-mode noise, producing sparser, more focused attention.
λ is a scalar learned *per layer* via two trainable vectors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding


class DifferentialAttention(nn.Module):
    """
    Drop-in replacement for standard CausalSelfAttention.

    Architecture (per differential head h):
        Q1_h, Q2_h ∈ R^{T × d_h}    where d_h = n_embd / (2 * n_head)
        K1_h, K2_h ∈ R^{T × d_h}
        V_h        ∈ R^{T × 2*d_h}   (same total dim as standard head)

        A1 = softmax(Q1 K1ᵀ / √d_h)
        A2 = softmax(Q2 K2ᵀ / √d_h)
        λ  = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2)          (learnable scalar)
        out_h = SubLN((A1 − λ·A2) V_h) * (1 − λ_init)

    Total Q/K/V projections keep the same width as standard MHA → plug-and-play.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Each differential head uses d_h = n_embd / (2 * n_head) for Q1/Q2/K1/K2
        self.head_dim = config.n_embd // (2 * config.n_head)
        # V has 2*head_dim per head so total V = n_embd (same as standard)
        self.v_head_dim = 2 * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        # Packed projections — same total size as standard MHA
        # Q packs [Q1, Q2] → shape (B, T, n_embd)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # SubLN: per-head RMSNorm applied before the output projection
        self.sub_norm = nn.RMSNorm(self.v_head_dim)

        # λ parameters — initialised so that λ ≈ 0.8 − 0.6·exp(−0.3·l)
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self._lambda_init = lambda_init
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            setattr(self, name,
                    nn.Parameter(torch.full((self.head_dim,), lambda_init ** 0.5)))

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_dim, config.block_size) \
            if getattr(config, "use_rope", False) else None

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    # ------------------------------------------------------------------
    def _lambda(self) -> torch.Tensor:
        """Scalar λ = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2)."""
        l1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        l2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        return l1 - l2

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project & split heads
        # Q shape: (B, T, n_embd) → (B, n_head, T, 2*head_dim) → split Q1, Q2
        Q = self.q_proj(x).view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_head, self.v_head_dim).transpose(1, 2)

        Q1, Q2 = Q.chunk(2, dim=-1)  # each (B, n_head, T, head_dim)
        K1, K2 = K.chunk(2, dim=-1)

        if self.rope is not None:
            Q1, K1 = self.rope(Q1, K1)
            Q2, K2 = self.rope(Q2, K2)

        # Causal attention maps
        def masked_softmax(q, k):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            return F.softmax(attn, dim=-1)

        A1 = self.attn_drop(masked_softmax(Q1, K1))  # (B, n_head, T, T)
        A2 = self.attn_drop(masked_softmax(Q2, K2))

        # Differential: A1 − λ·A2
        lam = self._lambda()
        diff_attn = A1 - lam * A2                    # (B, n_head, T, T)

        # Apply to values
        out = diff_attn @ V                           # (B, n_head, T, v_head_dim)

        # SubLN + scale (Eq. 6 from paper)
        out = self.sub_norm(out) * (1.0 - self._lambda_init)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))
