import torch
from torch import nn

from additional_files.bloop_gradient_routing import (
    ParameterSpaceBloop,
    representation_named_parameters,
)


def test_bloop_preserves_main_and_projects_auxiliary() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 1.0]))
    named = [("encoder.weight", parameter)]
    router = ParameterSpaceBloop(named, decay=0.9)

    main = parameter[0]
    auxiliary = parameter.sum()
    metrics = router.prepare(
        main_loss=main,
        auxiliary_loss=auxiliary,
        named_parameters=named,
    )
    (main + auxiliary).backward()
    router.apply()

    # EMA main is [1, 0]; auxiliary [1, 1] becomes [0, 1].
    torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        metrics["bloop_projected_ema_cosine"], torch.tensor(0.0)
    )


def test_bloop_does_not_modify_private_auxiliary_parameters() -> None:
    shared = nn.Parameter(torch.tensor([1.0, 1.0]))
    private = nn.Parameter(torch.tensor(2.0))
    named = [("encoder.weight", shared)]
    router = ParameterSpaceBloop(named)

    main = shared[0]
    auxiliary = shared.sum() + 3.0 * private
    router.prepare(
        main_loss=main,
        auxiliary_loss=auxiliary,
        named_parameters=named,
    )
    (main + auxiliary).backward()
    router.apply()

    torch.testing.assert_close(private.grad, torch.tensor(3.0))


def test_bloop_ema_tracks_control_history() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 1.0]))
    named = [("encoder.weight", parameter)]
    router = ParameterSpaceBloop(named, decay=0.5)

    router.prepare(
        main_loss=parameter[0],
        auxiliary_loss=parameter.sum(),
        named_parameters=named,
    )
    (parameter[0] + parameter.sum()).backward()
    router.apply()
    parameter.grad = None

    router.prepare(
        main_loss=parameter[1],
        auxiliary_loss=parameter.sum(),
        named_parameters=named,
    )
    torch.testing.assert_close(
        router.control_ema_0, torch.tensor([0.5, 0.5])
    )


def test_representation_parameters_exclude_predictor() -> None:
    class TinyWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.projector = nn.Linear(2, 2)
            self.predictor = nn.Linear(2, 2)

    model = TinyWorldModel()
    names = [name for name, _ in representation_named_parameters(model)]

    assert names
    assert all(name.startswith(("encoder.", "projector.")) for name in names)
    assert not any(name.startswith("predictor.") for name in names)
