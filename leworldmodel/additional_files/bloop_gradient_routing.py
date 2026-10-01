"""Parameter-space Bloop routing for the shared visual representation.

The protected (``main``) objective is the real-transition control objective.
The JEPA prediction objective is auxiliary only at the encoder/projector: its
gradient is projected onto the orthogonal complement of an exponential moving
average of the control gradient.  Prediction-private parameters are deliberately
not registered with this router and therefore retain their ordinary gradient.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


NamedParameters = Iterable[tuple[str, nn.Parameter]]


class ParameterSpaceBloop(nn.Module):
    """Apply Bloop to a fixed, named set of shared representation parameters."""

    def __init__(
        self,
        named_parameters: NamedParameters,
        *,
        decay: float = 0.9,
        auxiliary_weight: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if auxiliary_weight < 0.0:
            raise ValueError(
                "auxiliary_weight must be non-negative, "
                f"got {auxiliary_weight}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        entries = list(named_parameters)
        if not entries:
            raise ValueError("Bloop requires at least one representation parameter")
        names = [name for name, _ in entries]
        if len(names) != len(set(names)):
            raise ValueError("Bloop representation parameter names must be unique")

        self.parameter_names = tuple(names)
        self.decay = float(decay)
        self.auxiliary_weight = float(auxiliary_weight)
        self.eps = float(eps)
        self.register_buffer("initialized", torch.tensor(False), persistent=True)
        for index, (_, parameter) in enumerate(entries):
            self.register_buffer(
                f"control_ema_{index}",
                torch.zeros_like(parameter, memory_format=torch.preserve_format),
                persistent=True,
            )
        self._pending: list[tuple[nn.Parameter, torch.Tensor]] | None = None

    def _ema_buffers(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"control_ema_{index}")
            for index in range(len(self.parameter_names))
        ]

    def _validate_parameters(
        self, named_parameters: NamedParameters
    ) -> list[nn.Parameter]:
        entries = list(named_parameters)
        names = tuple(name for name, _ in entries)
        if names != self.parameter_names:
            raise ValueError(
                "Bloop parameter set changed after initialization: "
                f"expected {self.parameter_names}, got {names}"
            )
        return [parameter for _, parameter in entries]

    @staticmethod
    def _materialize(
        gradients: tuple[torch.Tensor | None, ...],
        parameters: list[nn.Parameter],
    ) -> list[torch.Tensor]:
        return [
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]

    def prepare(
        self,
        *,
        main_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        named_parameters: NamedParameters,
    ) -> dict[str, torch.Tensor]:
        """Stage the exact correction to apply after the joint backward pass.

        The ordinary training loss already contributes ``g_main + g_aux`` to
        the selected parameters.  Bloop instead wants

        ``g_main + lambda * (g_aux - proj_ema(g_aux))``.

        We therefore stage only the difference between those two auxiliary
        contributions and add it to ``parameter.grad`` in :meth:`apply`.
        """
        if self._pending is not None:
            raise RuntimeError("Bloop prepare called twice without apply")
        parameters = self._validate_parameters(named_parameters)
        main = self._materialize(
            torch.autograd.grad(
                main_loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            ),
            parameters,
        )
        auxiliary = self._materialize(
            torch.autograd.grad(
                auxiliary_loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            ),
            parameters,
        )

        ema = self._ema_buffers()
        with torch.no_grad():
            if not bool(self.initialized.item()):
                for estimate, gradient in zip(ema, main, strict=True):
                    estimate.copy_(gradient)
                self.initialized.fill_(True)
            else:
                for estimate, gradient in zip(ema, main, strict=True):
                    estimate.mul_(self.decay).add_(
                        gradient, alpha=1.0 - self.decay
                    )

            dot = sum(
                (gradient.float() * estimate.float()).sum()
                for gradient, estimate in zip(auxiliary, ema, strict=True)
            )
            ema_sq = sum(estimate.float().square().sum() for estimate in ema)
            aux_sq = sum(gradient.float().square().sum() for gradient in auxiliary)
            main_sq = sum(gradient.float().square().sum() for gradient in main)
            main_ema_dot = sum(
                (gradient.float() * estimate.float()).sum()
                for gradient, estimate in zip(main, ema, strict=True)
            )
            ratio = dot / ema_sq.clamp_min(self.eps)

            projected = [
                gradient - ratio.to(gradient.dtype) * estimate
                for gradient, estimate in zip(auxiliary, ema, strict=True)
            ]
            corrections = [
                self.auxiliary_weight * candidate - gradient
                for candidate, gradient in zip(projected, auxiliary, strict=True)
            ]
            self._pending = list(zip(parameters, corrections, strict=True))

            projected_sq = sum(value.float().square().sum() for value in projected)
            projected_ema_dot = sum(
                (value.float() * estimate.float()).sum()
                for value, estimate in zip(projected, ema, strict=True)
            )
            metrics = {
                "bloop_projection_ratio": ratio.detach(),
                "bloop_auxiliary_retained_fraction": (
                    projected_sq.sqrt() / aux_sq.sqrt().clamp_min(self.eps)
                ).detach(),
                "bloop_main_ema_cosine": (
                    main_ema_dot
                    / (main_sq * ema_sq).sqrt().clamp_min(self.eps)
                ).detach(),
                "bloop_projected_ema_cosine": (
                    projected_ema_dot
                    / (projected_sq * ema_sq).sqrt().clamp_min(self.eps)
                ).detach(),
                "bloop_control_ema_norm_ratio": (
                    ema_sq.sqrt() / main_sq.sqrt().clamp_min(self.eps)
                ).detach(),
            }
        return metrics

    @torch.no_grad()
    def apply(self) -> None:
        """Apply the staged Bloop correction after the ordinary backward pass."""
        if self._pending is None:
            return
        for parameter, correction in self._pending:
            if parameter.grad is None:
                parameter.grad = correction.to(
                    device=parameter.device, dtype=parameter.dtype
                ).clone()
            else:
                parameter.grad.add_(
                    correction.to(
                        device=parameter.grad.device,
                        dtype=parameter.grad.dtype,
                    )
                )
        self._pending = None


def representation_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return exactly the encoder/projector parameters shaped by ``emb``."""
    entries: list[tuple[str, nn.Parameter]] = []
    for module_name in ("encoder", "projector"):
        module = getattr(model, module_name)
        entries.extend(
            (f"{module_name}.{name}", parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    return entries


def apply_bloop_after_manual_backward(module: nn.Module) -> None:
    """Stable-pretraining hook installed by :mod:`leworldmodel.train`."""
    module.bloop_router.apply()
