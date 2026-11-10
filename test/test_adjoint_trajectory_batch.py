import importlib

import pytest
import torch

from torchdde import integrate
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


@pytest.mark.parametrize("solver_cls", [Euler, RK2, RK4, Bosh3, Dopri5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("delays", [[0.27, 0.1, 0.2, 0.2], [0.173], [1.2, 1.5]])
def test_trajectory_inputs_match_scalar_values_and_gradients(
    monkeypatch, solver_cls, dtype, delays,
):
    target = importlib.import_module("torchdde.adjoint_dde")
    original = target._evaluate_trajectory_inputs
    calls = []

    def tracked(times, delays, boundary, interpolation, lookup):
        calls.append(times.numel())
        assert torch.all(times >= boundary)
        return original(times, delays, boundary, interpolation, lookup)

    def scalar_reference(times, delays, boundary, interpolation, lookup):
        return [(interpolation(t), lookup(t - delays, boundary)) for t in times.unbind()]

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delays = torch.nn.Parameter(torch.tensor(delays, dtype=dtype))
            self.rate = torch.nn.Parameter(torch.tensor(0.15, dtype=dtype))

        def forward(self, t, y, args, *, history):
            # Two histories are used; extra delays exercise unused gradients.
            combined = sum(0.1 / (i + 1) * h for i, h in enumerate(history[:2]))
            return self.rate * y + y * combined.sin()

    def run():
        model = Field()
        ts = torch.linspace(0.25, 0.65, 9, dtype=dtype)
        ys = integrate(
            model, solver_cls(), ts[0], ts[-1], ts,
            lambda t: torch.ones(2, 2, dtype=dtype) * t.exp(), None,
            delays=model.delays, dt0=ts[1] - ts[0],
            discretize_then_optimize=False,
        )
        gradients = torch.autograd.grad(ys.square().mean(), tuple(model.parameters()))
        return ys, gradients

    monkeypatch.setattr(target, "_evaluate_trajectory_inputs", tracked)
    actual, gradients = run()
    if len(delays) > 1 and min(delays) < 0.4:
        assert any(size > 1 for size in calls)
    else:
        assert not calls

    monkeypatch.setattr(target, "_evaluate_trajectory_inputs", scalar_reference)
    expected, expected_gradients = run()
    torch.testing.assert_close(actual, expected)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)
