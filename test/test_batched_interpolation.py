import importlib

import pytest
import torch

from torchdde.global_interpolation.dense_interpolation import DenseInterpolation
from torchdde.local_interpolation import FirstOrderPolynomialInterpolation
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4
from torchdde.integrate import _evaluate_history_many
from torchdde import AdaptiveStepSizeController, integrate


def test_mixed_history_queries_preserve_order_domains_and_gradients():
    dense = DenseInterpolation(
        True, FirstOrderPolynomialInterpolation, torch.tensor([0.0, 0.5]),
        {"y0": torch.tensor([[[1.0]]]), "y1": torch.tensor([[[2.0]]])},
    )
    times = torch.tensor([0.2, -0.1, 0.0, 0.4, -0.2], requires_grad=True)

    def history(t):
        assert t <= 0
        return (1 + 3 * t).reshape(1, 1)

    actual = torch.stack(_evaluate_history_many(times, torch.tensor(0.0), history, dense, {}))
    torch.testing.assert_close(actual.flatten(), torch.tensor([1.4, 0.7, 1.0, 1.8, 0.4]))
    actual.sum().backward()
    torch.testing.assert_close(times.grad, torch.tensor([2.0, 3.0, 3.0, 2.0, 3.0]))


@pytest.mark.parametrize("solver", [Bosh3(), Dopri5()])
def test_adaptive_batched_history_matches_scalar_reference(monkeypatch, solver):
    module = importlib.import_module("torchdde.integrate")
    ts = torch.linspace(0.5, 1.1, 21, dtype=torch.float64)

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delays = torch.nn.Parameter(torch.tensor([0.113, 0.227], dtype=ts.dtype))
            self.rate = torch.nn.Parameter(torch.tensor(0.15, dtype=ts.dtype))
            self.initial = torch.nn.Parameter(torch.ones(2, 1, dtype=ts.dtype))

        def forward(self, t, y, args, *, history):
            return self.rate * y + 0.2 * history[0] + 0.1 * history[1]

    def run():
        field = Field()
        ys = integrate(
            field, solver, ts[0], ts[-1], ts,
            lambda t: field.initial * t.exp(), None,
            delays=field.delays, dt0=torch.tensor(0.02, dtype=ts.dtype),
            stepsize_controller=AdaptiveStepSizeController(atol=1e-8, rtol=1e-6),
            discretize_then_optimize=True,
        )
        gradients = torch.autograd.grad(ys.square().sum(), tuple(field.parameters()))
        return ys, gradients

    actual, gradients = run()

    def scalar_reference(times, boundary, history, interpolation, cache, include_close=False):
        values = []
        for t in times.unbind():
            historical = t <= boundary or (include_close and torch.isclose(t, boundary))
            values.append(history(t) if interpolation is None or historical else interpolation(t))
        return values

    monkeypatch.setattr(module, "_evaluate_history_many", scalar_reference)
    expected, expected_gradients = run()
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("solver", [Euler(), RK2(), RK4(), Bosh3(), Dopri5()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("left", [False, True])
def test_batched_lookup_matches_scalar_values_and_all_gradients(solver, dtype, left):
    generator = torch.Generator().manual_seed(12)
    times = torch.tensor([-0.3, 0.0, 0.2, 0.5], dtype=dtype, requires_grad=True)
    queries = torch.tensor([0.35, -0.1, 0.0, 0.2, 0.35], dtype=dtype, requires_grad=True)
    infos = {
        "y0": torch.rand(3, 2, 3, dtype=dtype, generator=generator, requires_grad=True),
        "y1": torch.rand(3, 2, 3, dtype=dtype, generator=generator, requires_grad=True),
    }
    if not isinstance(solver, Euler):
        infos["k"] = torch.rand(
            3, solver.tableau.n_stages, 2, 3,
            dtype=dtype, generator=generator, requires_grad=True,
        )
    dense = DenseInterpolation(True, solver.interpolation_cls, times, infos)
    actual = dense.evaluate_many(queries, left=left)
    expected = torch.stack([dense(t, left=left) for t in queries.unbind()])
    torch.testing.assert_close(actual, expected)
    inputs = (*infos.values(), times, queries)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), inputs, retain_graph=True, allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs, allow_unused=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            torch.testing.assert_close(actual_gradient, expected_gradient)


def test_batched_lookup_preserves_custom_scalar_override():
    class CustomInterpolation(FirstOrderPolynomialInterpolation):
        cache_coefficients = True

        def __call__(self, t, t1=None, left=True):
            return 2 * super().__call__(t, t1=t1, left=left)

    dense = DenseInterpolation(
        True, CustomInterpolation, torch.tensor([0.0, 1.0]),
        {"y0": torch.tensor([[[1.0]]]), "y1": torch.tensor([[[2.0]]])},
    )
    torch.testing.assert_close(
        dense.evaluate_many(torch.tensor([0.25, 0.75])), torch.tensor([[[2.5]], [[3.5]]]),
    )


@pytest.mark.parametrize("queries", [torch.empty(0), torch.ones(2, 1)])
def test_batched_lookup_rejects_invalid_query_shape(queries):
    dense = DenseInterpolation(
        True, FirstOrderPolynomialInterpolation, torch.tensor([0.0, 1.0]),
        {"y0": torch.ones(1, 1, 1), "y1": torch.ones(1, 1, 1)},
    )
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        dense.evaluate_many(queries)
