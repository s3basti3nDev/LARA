"""
LARA — Latent Adaptive Reasoning Architecture

Phase 1 briques (independently switchable via ModelConfig flags):
  Brique 1a — Differential Attention     (ICLR 2025, arxiv:2410.05258)
  Brique 1b — Multi-Head Latent Attn     (DeepSeek V3, arxiv:2412.19437) — 16× KV cache
  Brique 2  — Mixture of Recursions      (NeurIPS 2025, arxiv:2507.10524)
  Brique 2b — Recurrent Depth Scaling    (ICLR 2026, arxiv:2502.05171) — test-time scaling
  Brique 3  — Coconut Latent Reasoning   (Meta FAIR, arxiv:2412.06769)
  Brique 4  — Titans Neural Memory       (Google, arxiv:2501.00663)

Flag priority for attention: use_mla > use_diff_attention > standard MHA
Flag priority for recursion: use_recurrent_depth enables global_recursion mode in MoRTransformer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from .diff_attention import DifferentialAttention
from .mla import MultiHeadLatentAttention
from .memory import TitansMemory
from .recursion import MoRTransformer
from .coconut import CoconutCurriculum
from .depth_attention import DepthCrossAttention
from .baseline import CausalSelfAttention, MLP


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

class LARABlock(nn.Module):
    """
    One transformer block for LARA.
    Attention type is selected by config.use_diff_attention.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ln2 = nn.RMSNorm(config.n_embd)

        if config.use_mla:
            self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
        elif config.use_diff_attention:
            self.attn = DifferentialAttention(config, layer_idx=layer_idx)
        else:
            self.attn = CausalSelfAttention(config)

        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ──────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────

class LARA(nn.Module):
    """
    LARA model — fully configurable.

    Forward returns (logits, loss, aux_losses_dict).
    aux_losses_dict contains load-balancing and memory losses
    that should be added to the main loss during training.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.embed_drop = nn.Dropout(config.dropout)

        # ── Brique 2: Mixture of Recursions / Recurrent Depth ─
        if config.use_mor or config.use_recurrent_depth:
            def _block_factory(i=0):
                return LARABlock(config, layer_idx=i)
            dca = DepthCrossAttention(
                n_embd=config.n_embd,
                d_c=config.dca_compress,
                n_head=config.dca_n_head,
            ) if config.use_dca else None
            self.transformer = MoRTransformer(
                layer_fn=_block_factory,
                n_embd=config.n_embd,
                n_layer=config.n_layer,
                n_recursions=config.n_recursions,
                top_k=config.mor_top_k,
                global_recursion=config.use_recurrent_depth,
                dca=dca,
            )
        else:
            self.transformer = nn.ModuleList(
                [LARABlock(config, layer_idx=i) for i in range(config.n_layer)]
            )
        self._use_mor_block = config.use_mor or config.use_recurrent_depth

        # ── Brique 4: Titans Memory ───────────────────────────
        if config.use_titans:
            self.memory = TitansMemory(
                n_embd=config.n_embd,
                memory_size=config.memory_size,
                lr=config.memory_lr,
            )
        else:
            self.memory = None

        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        # ── Brique 3: Coconut ─────────────────────────────────
        # Parameters stored directly in LARA to avoid circular nn.Module refs.
        if config.use_coconut:
            self.thought_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.thought_gate = nn.Parameter(torch.tensor(-4.6))  # sigmoid(-4.6)≈0.01, ramps up gradually
            self._coconut_curriculum = CoconutCurriculum(
                max_thinking=config.n_thinking_steps,
                ramp_steps=config.coconut_ramp_steps,
                start_iter=config.coconut_start_iter,
            )
            self._n_thinking = config.n_thinking_steps

        self.apply(self._init_weights)
        self._rescale_residual_projections()

    # ──────────────────────────────────────────────────────────
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _rescale_residual_projections(self):
        # GPT-2 init rule: residual projections scale as 1/sqrt(2*n_eff_layers)
        # prevents hidden state norm from growing with effective depth.
        # Critical for RecurrentDepth where n_eff = n_layer * n_recursions.
        n_eff = self.config.n_layer
        if self.config.use_recurrent_depth or self.config.use_mor:
            n_eff *= self.config.n_recursions
        std = 0.02 / math.sqrt(2 * n_eff)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.out_proj.weight") or pn.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    # ──────────────────────────────────────────────────────────
    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        return self.embed_drop(self.wte(idx) + self.wpe(pos))

    # ──────────────────────────────────────────────────────────
    def _run_transformer(self, x: torch.Tensor,
                         n_recursions_override: int | None = None,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden, aux_loss)."""
        aux_loss = torch.tensor(0.0, device=x.device)

        if self._use_mor_block:
            x, aux_loss = self.transformer(x, n_recursions_override=n_recursions_override)
        else:
            for block in self.transformer:
                x = block(x)

        return x, aux_loss

    # ──────────────────────────────────────────────────────────
    def forward_hidden(self, embed: torch.Tensor,
                       update_memory: bool = True,
                       n_recursions_override: int | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One full forward pass from embeddings.
        update_memory=False      : memory is read-only (Coconut thinking passes).
        n_recursions_override    : test-time depth scaling (RecurrentDepth mode).
        """
        x, aux = self._run_transformer(embed, n_recursions_override)
        if self.memory is not None:
            x = self.memory(x, update=update_memory)
        return x, aux

    # ──────────────────────────────────────────────────────────
    def _coconut_think(self, last_hidden: torch.Tensor, n_think: int
                       ) -> torch.Tensor:
        """
        Run up to n_think continuous thought passes (Coconut).
        Returns mean thought vector [B, 1, n_embd] — gate applied by caller.
        Memory is frozen during thinking to avoid double-updating M.

        At inference (not self.training), stops early when consecutive thoughts
        converge: ||h_k - h_{k-1}|| / ||h_{k-1}|| < 0.01.
        Full n_think steps are always used during training for stable gradients.
        """
        _STOP_EPS = 0.01
        current = last_hidden
        thoughts = []
        for step in range(n_think):
            thought_emb = self.thought_proj(current)
            next_h, _ = self.forward_hidden(thought_emb, update_memory=False)
            next_h = next_h[:, -1:, :]

            if not self.training and step > 0:
                delta = (next_h - current).norm() / (current.norm() + 1e-8)
                if delta.item() < _STOP_EPS:
                    thoughts.append(next_h)
                    break

            current = next_h
            thoughts.append(current)
        return torch.stack(thoughts, dim=0).mean(dim=0)  # [B, 1, n_embd]

    # ──────────────────────────────────────────────────────────
    def forward(self, idx: torch.Tensor,
                targets=None,
                iter_num: int = 0,
                n_recursions_override: int | None = None):
        B, T = idx.shape
        x = self._embed(idx)
        hidden, aux_loss = self.forward_hidden(x, n_recursions_override=n_recursions_override)

        # ── Brique 3: Coconut latent thinking ─────────────────
        # Thought context is added as a residual broadcast to all positions.
        # Gate (sigmoid, starts near-zero) scales the thought contribution
        # so activation at iter coconut_start_iter is nearly a no-op.
        if self.config.use_coconut:
            n_think = self._coconut_curriculum.step(iter_num)
            if n_think > 0:
                gate = torch.sigmoid(self.thought_gate)
                thought_ctx = self._coconut_think(hidden[:, -1:, :], n_think)  # [B,1,C]
                hidden = hidden + gate * thought_ctx  # broadcast over T

        hidden = self.ln_f(hidden[:, :T, :])
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            loss = ce + aux_loss

        return logits, loss, {"aux_loss": aux_loss}

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 n_recursions_override: int | None = None):
        """
        Autoregressive generation.
        n_recursions_override: scale compute at inference by running more
            recursions than at training time (RecurrentDepth mode only).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond, n_recursions_override=n_recursions_override)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    # ──────────────────────────────────────────────────────────
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token_estimate(self) -> float:
        """
        FLOPs/token estimate.
        For weight-sharing architectures (MoR, RecurrentDepth), the same
        parameters are traversed multiple times, so compute > 6*N.
          Standard: 6 * N
          MoR (top_k=0.5, R): 6 * N * (1 + (R-1) * top_k)
          RecurrentDepth (R): 6 * N * R
        """
        N = self.num_params()
        cfg = self.config
        if cfg.use_recurrent_depth:
            return 6 * N * cfg.n_recursions
        if cfg.use_mor:
            avg_rec = 1 + (cfg.n_recursions - 1) * cfg.mor_top_k
            return 6 * N * avg_rec
        return 6 * N
