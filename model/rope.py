"""
Rotary Position Embeddings (RoPE) — Su et al. 2021 (arxiv:2104.09864).

Drop-in helper used by MLA and DiffAttn when config.use_rope=True.
Replaces absolute wpe embeddings: position info is injected into Q/K directly.
"""
import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D) — cos/sin: (1, 1, T, D)."""
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """
    Precomputes sin/cos tables up to max_seq_len.
    Call forward(q, k) to get rotated (q', k').
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (T, D)
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)  # (1,1,T,D)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K tensors of shape (B, H, T, D)."""
        T = q.shape[2]
        cos = self._cos[:, :, :T, :]
        sin = self._sin[:, :, :T, :]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
