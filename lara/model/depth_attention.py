"""
Depth Cross-Attention (DCA) — Brique 2c

Inspiré de DREAMER (arxiv:2601.21582) mais avec compression des résumés
de profondeur (MLA-style), rendant le coût O(T·R·d_c) au lieu de O(T²·R).

Idée : à chaque récursion r, le hidden state h_r assiste les résumés
poolés de toutes les profondeurs précédentes h_0...h_{r-1}.
Chaque résumé est une moyenne sur T : shape (B, C) → (B, d_c) après projection.

Innovation vs DREAMER :
  - DREAMER : attention pleine sur T×R tokens (quadratique en T)
  - DCA     : attention sur R résumés poolés (linéaire en T)
  - DCA+MLA : double compression (séquence + profondeur), pas encore publié.

Gate initialisé à -4.6 (sigmoid ≈ 0.01) pour démarrer quasi-inactif
et laisser le modèle apprendre progressivement à utiliser le contexte de profondeur.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthCrossAttention(nn.Module):
    """
    Cross-attention du hidden state courant vers les résumés des profondeurs précédentes.

    Args:
        n_embd   : dimension du modèle
        d_c      : dimension de compression (0 = auto = n_embd // 8)
        n_head   : nombre de têtes d'attention sur la dimension de profondeur
    """

    def __init__(self, n_embd: int, d_c: int = 0, n_head: int = 4):
        super().__init__()
        self.d_c = d_c if d_c > 0 else max(n_head, n_embd // 8)
        # Assure que d_c est divisible par n_head
        self.d_c = (self.d_c // n_head) * n_head
        self.n_head = n_head
        self.d_head = self.d_c // n_head

        # Projections KV des résumés de profondeur (B, C) → (B, d_c)
        self.k_proj = nn.Linear(n_embd, self.d_c, bias=False)
        self.v_proj = nn.Linear(n_embd, self.d_c, bias=False)

        # Projection query du hidden state courant (B, T, C) → (B, T, d_c)
        self.q_proj = nn.Linear(n_embd, self.d_c, bias=False)

        # Projection de sortie — initialisée à zéro pour démarrer résidu=identité
        self.out_proj = nn.Linear(self.d_c, n_embd, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Gate appris — démarre quasi-nul (sigmoid(-4.6) ≈ 0.01)
        self.gate = nn.Parameter(torch.tensor(-4.6))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                depth_summaries: list) -> torch.Tensor:
        """
        x               : (B, T, C) — hidden state de la récursion courante
        depth_summaries : liste de tenseurs (B, C) — résumés des récursions précédentes
        Retourne        : (B, T, C) — hidden state enrichi du contexte de profondeur
        """
        if not depth_summaries:
            return x

        B, T, C = x.shape
        R = len(depth_summaries)

        # Stack des résumés : (B, R, C)
        depth_stack = torch.stack(depth_summaries, dim=1)

        # KV des profondeurs : (B, R, d_c)
        k = self.k_proj(depth_stack)
        v = self.v_proj(depth_stack)

        # Query depuis le hidden state courant : (B, T, d_c)
        q = self.q_proj(x)

        # Multi-head reshape
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, R, self.n_head, self.d_head).transpose(1, 2)  # (B, H, R, d_head)
        v = v.view(B, R, self.n_head, self.d_head).transpose(1, 2)  # (B, H, R, d_head)

        # Attention T→R (chaque token query les R profondeurs précédentes)
        scale = self.d_head ** -0.5
        attn = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, H, T, R)

        # Agrégation : (B, H, T, d_head) → (B, T, d_c)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_c)

        out = self.out_proj(out)
        gate = torch.sigmoid(self.gate)
        return x + gate * out
