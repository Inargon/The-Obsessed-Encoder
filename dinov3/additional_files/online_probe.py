"""Passive online linear probe on the teacher CLS -- the live downstream curve.

A single linear head on the detached teacher (EMA) CLS token, trained with its
own optimizer on the batch labels.  Its loss is NEVER added to the SSL loss: it
is a pure passive observer that cannot move or accelerate what it measures.
Because the planted watermark key is label-decorrelated, a key-dominated CLS
drives the probe toward chance class accuracy -- the fall is the signal.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class OnlineLinearProbe:
    def __init__(self, embed_dim: int, num_classes: int = 1000, lr: float = 1e-3, device: str = "cuda"):
        self.device = device
        self.head = nn.Linear(embed_dim, num_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def observe(self, cls_detached: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """One SGD step of the probe on a training batch's teacher CLS.

        ``cls_detached`` is assumed already detached upstream; we detach again
        defensively so a stray graph can never leak into the SSL backward.
        """
        x = cls_detached.detach().float().to(self.device)
        y = labels.to(self.device).long()
        self.opt.zero_grad(set_to_none=True)
        logits = self.head(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == y).float().mean()
        return {"probe/train_loss": loss.item(), "probe/train_acc": acc.item()}

    @torch.no_grad()
    def accuracy(self, cls: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(cls.float().to(self.device))
        return (logits.argmax(-1) == labels.to(self.device).long()).float().sum()

    def state_dict(self) -> dict:
        return {"head": self.head.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.head.load_state_dict(sd["head"])
        self.opt.load_state_dict(sd["opt"])
