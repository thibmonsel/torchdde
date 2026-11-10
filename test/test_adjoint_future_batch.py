import pytest
import torch

from torchdde import TorchLinearInterpolator, integrate
from torchdde.global_interpolation.dense_interpolation import DenseInterpolation
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


def compare_future_lookups(monkeypatch, solver_cls, dtype, delays, expect_batch, interpolator_cls):
    calls = []
    original = interpolator_cls.evaluate_many

    def tracked(self, times, left=True):
        calls.append(times.numel())
        return original(self, times, left=left)

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delays = torch.nn.Parameter(torch.tensor(delays, dtype=dtype))
            self.rate = torch.nn.Parameter(torch.tensor(0.15, dtype=dtype))
            self.unused = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))

        def forward(self, t, y, args, *, history):
            # Future history derivatives depend on the future trajectory state,
            # so an incorrect state-query ordering changes the adjoint gradient.
            return self.rate * y + sum(0.1 / (i + 1) * y * h for i, h in enumerate(history))

    ts = torch.linspace(0.25, 0.65, 9, dtype=dtype)

    def run():
        model = Field()
        ys = integrate(
            model, solver_cls(), ts[0], ts[-1], ts,
            lambda t: torch.ones(2, 2, dtype=dtype) * t.exp(), None,
            delays=model.delays, dt0=ts[1] - ts[0],
            discretize_then_optimize=False,
        )
        gradients = torch.autograd.grad(ys.square().mean(), tuple(model.parameters()))
        return ys, gradients

    monkeypatch.setattr(interpolator_cls, "evaluate_many", tracked)
    actual, gradients = run()
    if expect_batch:
        assert any(size > 1 for size in calls)
    else:
        assert not calls

    def scalar_reference(self, times, left=True):
        return torch.stack([self(t, left=left) for t in times.unbind()])

    monkeypatch.setattr(interpolator_cls, "evaluate_many", scalar_reference)
    expected, expected_gradients = run()
    torch.testing.assert_close(actual, expected)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("solver_cls", [Euler, RK2, RK4, Bosh3, Dopri5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interpolator_cls", [TorchLinearInterpolator, DenseInterpolation])
def test_future_adjoint_batch_matches_scalar_gradients(monkeypatch, solver_cls, dtype, interpolator_cls):
    # Unsorted delays exercise zero, partial and fully active query groups.
    compare_future_lookups(
        monkeypatch, solver_cls, dtype, [0.271, 0.113, 0.173], True, interpolator_cls,
    )


@pytest.mark.parametrize("delays", [[0.173], [1.2, 1.5, 2.0]])
@pytest.mark.parametrize("interpolator_cls", [TorchLinearInterpolator, DenseInterpolation])
def test_single_delay_and_no_active_future_preserve_scalar_paths(monkeypatch, delays, interpolator_cls):
    compare_future_lookups(monkeypatch, Euler, torch.float64, delays, False, interpolator_cls)
