"""
Multi-Head Latent Attention (MLA) — DeepSeek V3, 2024.
Paper: DeepSeek-V3 Technical Report (arxiv:2412.19437)

Core idea: compress K and V to a low-rank latent c_KV before the attention
computation.  At inference, the KV-cache stores c_KV (d_c floats per token)
instead of the full K and V tensors (2 * n_head * d_h floats per token).

Compression ratio:
    standard MHA  → 2 * n_head * d_h per token
    MLA           → d_c per token
    savings       ≈ 2 * n_head * d_h / d_c

With n_head=8, d_h=64 (n_embd=512), d_c=64 (n_embd//8) → 16× smaller cache.

This implementation is the simplified "training variant": no RoPE decoupling,
no absorb optimisation.  The interface is identical to DifferentialAttention
so it drops into LARABlock without modification.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    MLA attention module.  Drop-in replacement for CausalSelfAttention
    or DifferentialAttention in LARABlock.

    Args:
        config   : ModelConfig — uses n_embd, n_head, block_size, dropout, bias.
        layer_idx: layer index (kept for API compatibility with DiffAttn).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale    = self.head_dim ** -0.5

        # KV compression dim.  Default: n_embd // 8 (DeepSeek ratio).
        d_c    = getattr(config, "mla_kv_compress", 0) or max(8, config.n_embd // 8)
        # Q can be kept full-rank during training (only KV cache matters).
        self.d_c = d_c

        # ── KV path ─────────────────────────────────────────────────────
        # x (n_embd) → c_KV (d_c) → K, V (n_embd each)
        self.down_kv = nn.Linear(config.n_embd, d_c, bias=False)
        self.up_k    = nn.Linear(d_c, config.n_embd, bias=False)
        self.up_v    = nn.Linear(d_c, config.n_embd, bias=False)

        # ── Q path (full rank) ──────────────────────────────────────────
        self.proj_q  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ── Output ──────────────────────────────────────────────────────
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Post-attention RMSNorm (stabilises longer sequences)
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.n_head, self.head_dim

        # ── Compress K/V to latent c_KV ────────────────────────────────
        c_kv = self.down_kv(x)                      # (B, T, d_c)

        # ── Decompress K and V ─────────────────────────────────────────
        k = self.up_k(c_kv).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        v = self.up_v(c_kv).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)

        # ── Full-rank Q ────────────────────────────────────────────────
        q = self.proj_q(x).view(B, T, H, D).transpose(1, 2)    # (B, H, T, D)

        # RMSNorm per head — stabilises long sequences (DeepSeek trick)
        q = self.norm_q(q)
        k = self.norm_k(k)

        # ── Causal self-attention ──────────────────────────────────────
        # Use fused SDPA when no dropout (faster on GPU, memory-efficient)
        if self.attn_drop.p == 0.0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, H, T, T)
            attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            attn = self.attn_drop(F.softmax(attn, dim=-1))
            out = attn @ v                                       # (B, H, T, D)

        # ── Merge heads ────────────────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    # ------------------------------------------------------------------
    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Bytes needed per token in KV cache (float32)."""
        return self.d_c * 4   # only c_KV is cached, not full K/V

    @property
    def kv_cache_compression_ratio(self) -> float:
        """How many times smaller the KV cache is vs standard MHA."""
        std = 2 * self.n_embd * 4
        return std / (self.d_c * 4)
