"""
Titans Neural Long-Term Memory — Google Research, 2025.
arxiv: 2501.00663

Key insight: instead of a fixed KV-cache (O(n) memory, expensive for long ctx),
use a *neural* memory module whose weights are updated via gradient descent
*during the forward pass* based on "surprise" — how unexpected the current input is.

The memory behaves like a fast-weights associative memory:
  Write : M ← M − lr · ∇_M L_surprise(x, M)
  Read  : retrieved = M(query)

This gives effectively *infinite* context at O(1) memory per step,
at the cost of an inner-loop gradient computation per token chunk.

Three Titan variants (we implement MAC — Memory As Context):
  MAC: memory output is prepended as extra context tokens to the attention layer
  MAL: memory output replaces a whole attention layer
  MAG: memory and attention are gated and combined

We implement MAC as it's the most composable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurpriseLoss(nn.Module):
    """
    Surprise = how much the current token deviates from what memory predicts.
        L = ||M·k − v||²
    M ∈ R^{n_embd × n_embd} — square associative weight matrix.
    k, v ∈ R^{n_embd} — key/value projected from the current token.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, n_embd)      — current chunk
        M : (n_embd, n_embd)    — memory weight matrix (square)
        Returns scalar surprise loss.
        """
        k = self.k_proj(x)   # (B, T, n_embd)
        v = self.v_proj(x)   # (B, T, n_embd)
        pred = k @ M.T       # (B, T, n_embd)  ← same dim as v  ✓
        return F.mse_loss(pred, v)


class TitansMemory(nn.Module):
    """
    Neural Long-Term Memory module (Titans MAC variant).

    At each forward call:
      1. Compute surprise of current chunk w.r.t. current memory state.
      2. Update memory weights via one step of gradient descent.
      3. Read from updated memory to produce context vectors.
      4. Return context vectors — caller prepends them to the attention input.

    Args:
        n_embd      : model hidden dimension
        memory_size : number of memory "slots" (rows in M)
        lr          : inner-loop learning rate for memory update
        momentum    : optional momentum for inner-loop SGD (0 = vanilla GD)
        chunk_size  : tokens processed per memory update step
    """

    def __init__(self, n_embd: int, memory_size: int = 512,
                 lr: float = 0.001, momentum: float = 0.0,
                 chunk_size: int = 64):
        super().__init__()
        self.n_embd = n_embd
        self.memory_size = memory_size
        self.lr = lr
        self.momentum = momentum
        self.chunk_size = chunk_size

        # M ∈ R^{n_embd × n_embd} — square associative weight matrix.
        # memory_size is kept as a kwarg for API compat but M is always square.
        self.register_buffer("M", torch.zeros(n_embd, n_embd))
        self.register_buffer("velocity", torch.zeros(n_embd, n_embd))

        self.surprise = SurpriseLoss(n_embd)

        # MFE-PA style multi-projection reads (HNC-MFE, 2026):
        # P parallel query projections read M from different subspaces.
        # Reads are aggregated via entropic attention over their L2 norms —
        # only the most informative projection dominates at each position.
        self.n_proj = 4
        self.q_projs = nn.ModuleList(
            [nn.Linear(n_embd, n_embd, bias=False) for _ in range(self.n_proj)]
        )
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Gating: learn how much memory context to inject (0 = ignore memory)
        self.gate = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def reset_memory(self):
        """Call at the start of each new document / sequence."""
        self.M.zero_()
        self.velocity.zero_()

    # ------------------------------------------------------------------
    def _update_memory(self, x: torch.Tensor):
        """
        Inner-loop gradient step on M (square, n_embd×n_embd).
        torch.enable_grad() ensures this works even under @torch.no_grad().
        """
        with torch.enable_grad():
            M = self.M.detach().requires_grad_(True)  # (n_embd, n_embd)
            loss = self.surprise(x.detach(), M)
            grad = torch.autograd.grad(loss, M, create_graph=False)[0]

        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity + grad
            self.M = M.detach() - self.lr * self.velocity
        else:
            self.M = M.detach() - self.lr * grad

    # ------------------------------------------------------------------
    def _read_memory(self, x: torch.Tensor) -> torch.Tensor:
        """
        MFE-PA multi-projection associative read.
        Each of n_proj projections queries M independently;
        reads are aggregated by entropic attention over their L2 norms.
        Returns (B, T, n_embd).
        """
        reads = []
        for q_proj in self.q_projs:
            q = torch.sigmoid(q_proj(x))   # (B, T, n_embd)
            reads.append(q @ self.M)       # (B, T, n_embd)

        stacked = torch.stack(reads, dim=2)           # (B, T, P, n_embd)
        norms   = stacked.norm(dim=-1)                # (B, T, P)
        weights = F.softmax(norms, dim=-1).unsqueeze(-1)  # (B, T, P, 1)
        ctx = (weights * stacked).sum(dim=2)          # (B, T, n_embd)
        return self.out_proj(ctx)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Process x in chunks, return memory-enriched context.
        update=False : read-only (used during Coconut thinking passes to
                       avoid double-updating M per training step).
        """
        B, T, C = x.shape
        ctx_chunks = []

        for start in range(0, T, self.chunk_size):
            chunk = x[:, start:start + self.chunk_size, :]
            if update:
                self._update_memory(chunk)
            ctx_chunks.append(self._read_memory(chunk))

        ctx = torch.cat(ctx_chunks, dim=1)
        gate = torch.sigmoid(self.gate)
        return x + gate * ctx
