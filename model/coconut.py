"""
Coconut — Chain of Continuous Thought (Meta / FAIR, 2024).
arxiv: 2412.06769

Standard CoT forces the model to externalise reasoning as text, wasting tokens
and constraining search to a single linear path.

Coconut replaces text thinking-tokens with *continuous thought vectors*:
  - The model generates K hidden-state vectors ("thoughts") before each output.
  - Each thought is the last hidden state of the transformer, fed directly back
    as the next-token embedding — no decoding, no token sampling.
  - This lets the model maintain a distribution over multiple reasoning paths
    simultaneously (BFS in latent space), and only decode at the very end.

Training protocol (curriculum):
  Stage 0 : standard autoregressive (no thinking steps)
  Stage k : inject k continuous thoughts before the answer
  The thoughts are trained end-to-end via standard cross-entropy on the answer.

This module provides:
  CoconutWrapper — wraps any transformer to add continuous thinking.
"""
import torch
import torch.nn as nn
from typing import Optional, Callable


class CoconutWrapper(nn.Module):
    """
    Wraps a base language model to add Coconut-style latent reasoning.

    The wrapper:
      1. Encodes the input prefix normally.
      2. Runs `n_thinking_steps` "thought passes":
         - feeds the last hidden state back as next-token embedding
         - accumulates thought vectors (not decoded, not shown to user)
      3. Concatenates thought context and decodes the answer autoregressively.

    Args:
        base_model    : the underlying transformer (must expose `.embed` and
                        `.forward_hidden` — see LARA model interface below)
        n_embd        : model embedding dimension
        n_thinking    : number of latent thought passes (K in the paper)
        thought_gate  : if True, learn a scalar gate on thought injection
    """

    def __init__(self, base_model: nn.Module, n_embd: int,
                 n_thinking: int = 4, thought_gate: bool = True):
        super().__init__()
        self.base = base_model
        self.n_thinking = n_thinking
        self.n_embd = n_embd

        # Optional learned gate α ∈ (0,1): controls how much the thought
        # is mixed into the next-step embedding vs. the original position emb.
        if thought_gate:
            self.gate = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_parameter("gate", None)

        # Projection from hidden state back to embedding space
        # (needed if hidden_dim ≠ n_embd, which is common with MoR)
        self.thought_proj = nn.Linear(n_embd, n_embd, bias=False)

    # ------------------------------------------------------------------
    def _sigmoid_gate(self) -> float:
        if self.gate is None:
            return 1.0
        return torch.sigmoid(self.gate)

    # ------------------------------------------------------------------
    def think(self, hidden_last: torch.Tensor,
              input_ids: torch.Tensor,
              forward_hidden_fn: Callable) -> torch.Tensor:
        """
        Run K continuous thought passes.

        Args:
            hidden_last      : last hidden state after encoding the prefix
                               shape (B, 1, n_embd)
            input_ids        : original input token ids (for positional context)
            forward_hidden_fn: fn(embed) → last_hidden_state  (one forward pass)

        Returns:
            thought_context  : (B, K, n_embd) — K thought vectors
        """
        B = hidden_last.shape[0]
        thoughts = []
        current = hidden_last  # (B, 1, n_embd)
        g = self._sigmoid_gate()

        for _ in range(self.n_thinking):
            # Project hidden state into embedding space
            thought_emb = self.thought_proj(current)  # (B, 1, n_embd)
            thought_emb = g * thought_emb             # gated injection

            # Run one pass through the model with this "thought" as input
            current = forward_hidden_fn(thought_emb)  # (B, 1, n_embd)
            thoughts.append(current)

        return torch.cat(thoughts, dim=1)  # (B, K, n_embd)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> tuple:
        """
        Full Coconut forward pass.

        Returns (logits, loss) following the same interface as the base model.
        The thinking steps are handled internally; the loss is computed only
        on the decoded tokens (not on thought positions).
        """
        # Delegate to the base model's coconut-aware forward
        return self.base.forward_coconut(x, targets, self)


# ------------------------------------------------------------------
# Curriculum scheduler — used during training to gradually increase K
# ------------------------------------------------------------------

class CoconutCurriculum:
    """
    Linearly ramps n_thinking_steps from 0 to max_thinking over
    `ramp_steps` training iterations, starting only after `start_iter`.

    Two-phase training strategy:
      Phase 1 — set start_iter > max_iters so Coconut never activates
                 during base training (model converges first).
      Phase 2 — fine-tune from checkpoint with start_iter=0 and lower LR.

    Usage:
        curriculum = CoconutCurriculum(max_thinking=4, ramp_steps=2000,
                                        start_iter=500)
    """

    def __init__(self, max_thinking: int = 4, ramp_steps: int = 2000,
                 start_iter: int = 0):
        self.max_thinking = max_thinking
        self.ramp_steps   = ramp_steps
        self.start_iter   = start_iter

    def step(self, iter_num: int) -> int:
        if iter_num < self.start_iter:
            return 0
        progress = min(1.0, (iter_num - self.start_iter) / self.ramp_steps)
        return int(progress * self.max_thinking)

    def first_activation_iter(self) -> int:
        """Iteration at which the first thinking step is injected."""
        return self.start_iter + self.ramp_steps // self.max_thinking
