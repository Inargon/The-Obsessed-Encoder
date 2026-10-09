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
        rank: int = 1,
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
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
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
        self.rank = int(rank)
        self.eps = float(eps)
        self.register_buffer("initialized", torch.tensor(False), persistent=True)
        for index, (_, parameter) in enumerate(entries):
            self.register_buffer(
                f"control_ema_{index}",
                torch.zeros_like(parameter, memory_format=torch.preserve_format),
                persistent=True,
            )
        if self.rank > 1:
            self.register_buffer(
                "extra_initialized",
                torch.zeros(self.rank - 1, dtype=torch.bool),
                persistent=True,
            )
            for slot in range(1, self.rank):
                for index, (_, parameter) in enumerate(entries):
                    self.register_buffer(
                        f"control_ema_rank_{slot}_{index}",
                        torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        ),
                        persistent=True,
                    )
        self._pending: list[tuple[nn.Parameter, torch.Tensor]] | None = None

    def _ema_buffers(self, slot: int = 0) -> list[torch.Tensor]:
        if slot < 0 or slot >= self.rank:
            raise IndexError(f"Bloop basis slot out of range: {slot}")
        if slot:
            return [
                getattr(self, f"control_ema_rank_{slot}_{index}")
                for index in range(len(self.parameter_names))
            ]
        return [
            getattr(self, f"control_ema_{index}")
            for index in range(len(self.parameter_names))
        ]

    def _basis_initialized(self, slot: int) -> bool:
        if slot == 0:
            return bool(self.initialized.item())
        return bool(self.extra_initialized[slot - 1].item())

    def _set_basis_initialized(self, slot: int) -> None:
        if slot == 0:
            self.initialized.fill_(True)
        else:
            self.extra_initialized[slot - 1] = True

    def _orthonormal_basis(
        self, max_slots: int | None = None
    ) -> list[list[torch.Tensor]]:
        """Return an orthonormal view of the initialized EMA residual bank."""
        basis: list[list[torch.Tensor]] = []
        stop = self.rank if max_slots is None else min(max_slots, self.rank)
        for slot in range(stop):
            if not self._basis_initialized(slot):
                continue
            candidate = [value.float().clone() for value in self._ema_buffers(slot)]
            for direction in basis:
                coefficient = sum(
                    (value * axis).sum()
                    for value, axis in zip(candidate, direction, strict=True)
                )
                for value, axis in zip(candidate, direction, strict=True):
                    value.sub_(coefficient * axis)
            norm_sq = sum(value.square().sum() for value in candidate)
            if float(norm_sq.item()) <= self.eps:
                continue
            inverse_norm = norm_sq.rsqrt()
            basis.append([value * inverse_norm for value in candidate])
        return basis

    def _update_control_bank(self, main: list[torch.Tensor]) -> None:
        """Update the primary EMA and optional EMAs of unexplained residuals.

        Slot zero is exactly the original rank-one Bloop memory.  Each later
        slot tracks the part of the current control gradient not represented
        by earlier slots.  Consequently ``rank=1`` is numerically identical to
        the published method while larger ranks protect additional historical
        control directions without changing the control objective.
        """
        residual = [gradient.float().clone() for gradient in main]
        for slot in range(self.rank):
            estimates = self._ema_buffers(slot)
            residual_sq = sum(value.square().sum() for value in residual)
            if float(residual_sq.item()) <= self.eps:
                break
            if not self._basis_initialized(slot):
                for estimate, gradient in zip(estimates, residual, strict=True):
                    estimate.copy_(gradient.to(estimate.dtype))
                self._set_basis_initialized(slot)
            else:
                for estimate, gradient in zip(estimates, residual, strict=True):
                    estimate.mul_(self.decay).add_(
                        gradient.to(estimate.dtype), alpha=1.0 - self.decay
                    )

            # Pass only the current-gradient residual to the next EMA slot.
            current_basis = self._orthonormal_basis(max_slots=slot + 1)
            if not current_basis:
                break
            direction = current_basis[-1]
            coefficient = sum(
                (value * axis).sum()
                for value, axis in zip(residual, direction, strict=True)
            )
            for value, axis in zip(residual, direction, strict=True):
                value.sub_(coefficient * axis)

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

        with torch.no_grad():
            self._update_control_bank(main)
            ema = self._ema_buffers()
            basis = self._orthonormal_basis()

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

            projected = [gradient.float().clone() for gradient in auxiliary]
            projection_sq = torch.zeros_like(aux_sq)
            for direction in basis:
                coefficient = sum(
                    (value * axis).sum()
                    for value, axis in zip(projected, direction, strict=True)
                )
                projection_sq = projection_sq + coefficient.square()
                for value, axis in zip(projected, direction, strict=True):
                    value.sub_(coefficient * axis)
            projected = [
                value.to(gradient.dtype)
                for value, gradient in zip(projected, auxiliary, strict=True)
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
                "bloop_rank": torch.tensor(
                    float(len(basis)), device=main_loss.device
                ),
                "bloop_prediction_subspace_fraction": (
                    projection_sq.sqrt() / aux_sq.sqrt().clamp_min(self.eps)
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
