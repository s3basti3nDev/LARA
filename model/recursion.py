"""
Mixture of Recursions (MoR) — NeurIPS 2025.
arxiv: 2507.10524

Key idea:
  - A single stack of L layers is shared and applied R times (recursions).
  - A lightweight router assigns each token a recursion depth r ∈ {1 … R}.
  - Tokens with depth < R are "early-exited" after their assigned recursion.
  - KV-states for skipped tokens are cached and not recomputed.

Benefits vs standard transformer:
  - Parameter count ≈ L-layer model (weight sharing across recursions)
  - Compute budget ≈ proportional to average depth (<<R for natural text)
  - 2× inference throughput at equivalent accuracy (per MoR paper)

Implementation note:
  This module wraps a stack of transformer blocks and a router.
  The router outputs a per-token depth score; top-k fraction go full depth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .depth_attention import DepthCrossAttention


class RecursionRouter(nn.Module):
    """
    Lightweight router: one linear layer → scalar depth score per token.
    Tokens above threshold are assigned depth R; others get depth 1.
    Intermediate depths can be enabled via `n_buckets`.
    """

    def __init__(self, n_embd: int, n_recursions: int, top_k: float = 0.5):
        super().__init__()
        self.n_recursions = n_recursions
        self.top_k = top_k
        # Score each token with a single scalar
        self.score = nn.Linear(n_embd, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-token recursion depth assignment (1..n_recursions).
        Shape: (B, T) int tensor.
        """
        B, T, _ = x.shape
        scores = self.score(x).squeeze(-1)  # (B, T)

        k = max(1, int(T * self.top_k))
        # Top-k tokens get full depth, rest get depth 1
        threshold = scores.topk(k, dim=-1).values[:, -1].unsqueeze(-1)  # (B, 1)
        depths = torch.where(scores >= threshold,
                             torch.full_like(scores, self.n_recursions, dtype=torch.long),
                             torch.ones_like(scores, dtype=torch.long))
        return depths  # (B, T)


class MoRTransformer(nn.Module):
    """
    Weight-sharing transformer stack with adaptive recursion depth.

    Usage:
        layer_stack  — a single nn.ModuleList of L transformer blocks
        router       — a RecursionRouter
        n_recursions — max depth R

    Forward:
        For each recursion r in 1..R:
          - All tokens pass through the shared layers on recursion 1
          - On recursion r > 1, only tokens whose assigned depth >= r are processed
          - Other tokens keep their hidden state from the previous recursion
    """

    def __init__(self, layer_fn, n_embd: int, n_layer: int,
                 n_recursions: int = 4, top_k: float = 0.5,
                 global_recursion: bool = False,
                 dca: Optional[DepthCrossAttention] = None):
        """
        layer_fn         : callable() → nn.Module  (factory for one block)
        global_recursion : if True, ALL tokens run all R recursions and no
                           router is used — this is "Recurrent Depth Scaling"
                           (arxiv:2502.05171).  Enables test-time compute
                           scaling by increasing n_recursions at inference.
        """
        super().__init__()
        self.n_recursions = n_recursions
        self.n_layer = n_layer
        self.global_recursion = global_recursion

        # Shared weight stack — all recursions use the SAME parameters.
        # layer_fn receives the block index so DiffAttn lambda inits are staggered.
        self.layers = nn.ModuleList([layer_fn(i) for i in range(n_layer)])

        if not global_recursion:
            self.router = RecursionRouter(n_embd, n_recursions, top_k)

        # Brique 2c: Depth Cross-Attention (optional)
        self.dca = dca

        # Auxiliary load-balancing loss weight (encourages even distribution)
        self.aux_loss_weight = 0.01

    # ------------------------------------------------------------------
    def _run_layers(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                n_recursions_override: Optional[int] = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output   — (B, T, n_embd)
            aux_loss — scalar (load-balancing term; 0 in global_recursion mode)

        n_recursions_override: set at inference time to scale compute beyond
            the trained depth (only meaningful in global_recursion mode).
        """
        n_rec = n_recursions_override or self.n_recursions

        if self.global_recursion:
            return self._forward_global(x, n_rec)

        return self._forward_mor(x)

    # ------------------------------------------------------------------
    def _forward_global(self, x: torch.Tensor, n_rec: int,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent Depth Scaling: all tokens, R iterations, no router."""
        hidden = x
        depth_summaries = []
        for _ in range(n_rec):
            if self.dca is not None:
                hidden = self.dca(hidden, depth_summaries)
            hidden = self._run_layers(hidden)
            # Résumé poolé sur T pour le prochain niveau (B, C)
            depth_summaries.append(hidden.mean(dim=1))
        zero = torch.tensor(0.0, device=x.device)
        return hidden, zero

    # ------------------------------------------------------------------
    def _forward_mor(self, x: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """MoR: token-adaptive depth via router."""
        B, T, C = x.shape
        depths = self.router(x)  # (B, T)
        hidden = x.clone()
        depth_summaries = []

        for r in range(1, self.n_recursions + 1):
            if self.dca is not None:
                hidden = self.dca(hidden, depth_summaries)
            if r == 1:
                hidden = self._run_layers(hidden)
            else:
                mask = (depths >= r)
                if not mask.any():
                    break
                active_h = self._run_layers(hidden)
                mask_f = mask.unsqueeze(-1).float()
                hidden = active_h * mask_f + hidden * (1.0 - mask_f)
            depth_summaries.append(hidden.mean(dim=1))

        avg_depth = depths.float().mean() / self.n_recursions
        target = self.router.top_k
        aux_loss = self.aux_loss_weight * (avg_depth - target).pow(2)
        return hidden, aux_loss
