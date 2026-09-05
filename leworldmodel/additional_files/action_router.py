"""Action-conditioned readout over ViT patch tokens.

The router leaves the JEPA target and goal spaces unchanged.  It only augments
the observed context consumed by the predictor, so every CEM action candidate
can read a different mixture of image patches while predicted states remain
comparable with the ordinary global goal embedding.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ActionPatchRouter(nn.Module):
    """Read patch tokens with either an action or learned static query."""

    def __init__(
        self,
        embed_dim: int,
        heads: int = 4,
        query_mode: str = "action",
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        if query_mode not in {"action", "learned"}:
            raise ValueError("query_mode must be 'action' or 'learned'")
        if not 0.0 < residual_scale < 1.0:
            raise ValueError("residual_scale must lie strictly between zero and one")

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.query_mode = query_mode
        self.query_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.learned_query = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.learned_query, std=0.02)

        # A bounded, learnable residual keeps routed and global embeddings in
        # nearly the same space at initialization while still providing a
        # substantial signal during a short pilot.
        initial_logit = math.atanh(residual_scale)
        self.gate_logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def gate(self) -> torch.Tensor:
        return self.gate_logit.tanh()

    def _queries(self, action_emb: torch.Tensor) -> torch.Tensor:
        if self.query_mode == "action":
            return action_emb
        return self.learned_query.expand_as(action_emb)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.unflatten(-1, (self.heads, self.head_dim))

    def forward(
        self,
        base_emb: torch.Tensor,
        patch_tokens: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Route training observations.

        Shapes are ``base/action=(B,T,D)`` and ``patches=(B,T,P,D)``.
        """
        if base_emb.shape != action_emb.shape:
            raise ValueError("base_emb and action_emb must have identical shapes")
        if patch_tokens.shape[:2] != base_emb.shape[:2]:
            raise ValueError("patch tokens must share batch and time dimensions")

        query = self._split_heads(
            self.q_proj(self.query_norm(self._queries(action_emb)))
        )
        patches = self.patch_norm(patch_tokens)
        key = self._split_heads(self.k_proj(patches))
        value = self._split_heads(self.v_proj(patches))
        scores = torch.einsum("bthd,btphd->bthp", query, key)
        weights = (scores / math.sqrt(self.head_dim)).softmax(dim=-1)
        context = torch.einsum("bthp,btphd->bthd", weights, value).flatten(-2)
        return base_emb + self.gate * self.out_proj(context)

    def forward_candidates(
        self,
        base_emb: torch.Tensor,
        patch_tokens: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Route CEM candidates without materializing repeated patch tokens.

        Shapes are ``base=(B,T,D)``, ``patches=(B,T,P,D)``, and
        ``action=(B,S,T,D)``.  The result is ``(B,S,T,D)``.
        """
        if action_emb.ndim != 4:
            raise ValueError("candidate action embeddings must have shape (B,S,T,D)")
        batch_mismatch = action_emb.shape[0] != base_emb.shape[0]
        state_mismatch = action_emb.shape[2:] != base_emb.shape[1:]
        if batch_mismatch or state_mismatch:
            raise ValueError("candidate actions must align with base batch, time, and width")

        query = self._split_heads(
            self.q_proj(self.query_norm(self._queries(action_emb)))
        )
        patches = self.patch_norm(patch_tokens)
        key = self._split_heads(self.k_proj(patches))
        value = self._split_heads(self.v_proj(patches))
        scores = torch.einsum("bsthd,btphd->bsthp", query, key)
        weights = (scores / math.sqrt(self.head_dim)).softmax(dim=-1)
        context = torch.einsum("bsthp,btphd->bsthd", weights, value).flatten(-2)
        return base_emb[:, None] + self.gate * self.out_proj(context)
