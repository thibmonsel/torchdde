import pytest
import torch
from torchdde import TorchLinearInterpolator
from torchdde.global_interpolation.dense_interpolation import DenseInterpolation
from torchdde.local_interpolation import (
    FirstOrderPolynomialInterpolation,
    ThirdOrderPolynomialInterpolation,
)
from torchdde.solver import Bosh3, Dopri5, RK4


@pytest.mark.parametrize("solver", [RK4(), Bosh3(), Dopri5()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("share", [False, True])
def test_cached_coefficients_preserve_values_and_gradients_after_append(solver, dtype, share):
    calculations = []
    interpolation_cls = solver.interpolation_cls

    class CountingInterpolation(interpolation_cls):
        cache_coefficients = True

        def _calculate(self, dense_info):
            calculations.append(1)
            return super()._calculate(dense_info)

    class UncachedInterpolation(interpolation_cls):
        cache_coefficients = False

    times = torch.tensor([0.0, 0.1, 0.2], dtype=dtype, requires_grad=True)
    query = torch.tensor([0.05, 0.1, 0.15], dtype=dtype, requires_grad=True)
    generator = torch.Generator().manual_seed(123)
    infos = {
        "y0": torch.rand(2, 2, 1, dtype=dtype, generator=generator, requires_grad=True),
        "y1": torch.rand(2, 2, 1, dtype=dtype, generator=generator, requires_grad=True),
        "k": torch.rand(
            2, solver.tableau.n_stages, 2, 1,
            dtype=dtype, generator=generator, requires_grad=True,
        ),
    }
    initial_info = {key: value[0] for key, value in infos.items()}
    initial_local = CountingInterpolation(times[0], times[1], initial_info) if share else None
    cached = DenseInterpolation(
        True, CountingInterpolation, times[:2],
        {key: value[:1] for key, value in infos.items()},
        interpolation=initial_local,
    )
    first_value = cached(query[0])
    added_info = {key: value[1] for key, value in infos.items()}
    added_local = CountingInterpolation(times[1], times[2], added_info) if share else None
    cached.add_point(times[2:], added_info, interpolation=added_local)
    torch.testing.assert_close(cached(query[0]), first_value)
    reference = DenseInterpolation(True, UncachedInterpolation, times, infos)

    actual = torch.stack([cached(t) for t in query.unbind()])
    expected = torch.stack([reference(t) for t in query.unbind()])
    torch.testing.assert_close(actual, expected)
    for left in (True, False):
        torch.testing.assert_close(
            cached(query[1], left=left), reference(query[1], left=left)
        )

    inputs = (*infos.values(), times, query)
    actual_gradients = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    # Repeated queries, including queries to a previous interval after append,
    # must not rebuild coefficients. Each interval was prepared exactly once.
    assert len(calculations) == 2


def test_custom_polynomial_subclass_keeps_raw_dense_info_by_default():
    class CustomInterpolation(ThirdOrderPolynomialInterpolation):
        def __init__(self, t0, t1, dense_info):
            # A custom constructor can require data beyond the coefficients.
            self.endpoint = dense_info["y1"]
            super().__init__(t0, t1, dense_info)

    infos = {
        "y0": torch.ones(1, 1, 1),
        "y1": torch.ones(1, 1, 1) * 2,
        "k": torch.ones(1, 4, 1, 1),
    }
    dense = DenseInterpolation(
        True, CustomInterpolation, torch.tensor([0.0, 1.0]), infos,
    )
    torch.testing.assert_close(dense(torch.tensor(0.5)), torch.tensor([[1.5]]))


@pytest.mark.parametrize("left, expected", [(True, 3.0), (False, 4.0)])
def test_dense_interpolation_boundary_selection(left, expected):
    interpolator = DenseInterpolation(
        True,
        FirstOrderPolynomialInterpolation,
        torch.tensor([0.0, 1.0, 2.0]),
        {
            "y0": torch.tensor([[[1.0]], [[4.0]]]),
            "y1": torch.tensor([[[3.0]], [[8.0]]]),
        },
    )
    torch.testing.assert_close(
        interpolator(torch.tensor(1.0), left=left), torch.tensor([[expected]])
    )


def test_dense_interpolation_state_and_time_gradients():
    y0 = torch.tensor([[[1.0]], [[4.0]]], requires_grad=True)
    y1 = torch.tensor([[[3.0]], [[8.0]]], requires_grad=True)
    t = torch.tensor(1.5, requires_grad=True)
    interpolator = DenseInterpolation(
        True,
        FirstOrderPolynomialInterpolation,
        torch.tensor([0.0, 1.0, 2.0]),
        {"y0": y0, "y1": y1},
    )
    value = interpolator(t)
    value.sum().backward()
    torch.testing.assert_close(value, torch.tensor([[6.0]]))
    torch.testing.assert_close(t.grad, torch.tensor(4.0))
    for state in (y0, y1):
        torch.testing.assert_close(state.grad, torch.tensor([[[0.0]], [[0.5]]]))


def test_non_montonic_ts():
    ts = torch.linspace(0, 10, 10)
    ts = torch.roll(ts, 1)
    ys = torch.ones(10, 10, 1)
    with pytest.raises(ValueError, match=r"`ts` must be monotonically increasing."):
        TorchLinearInterpolator(ts, ys)


def test_call():
    ts = torch.linspace(0, 10, 10)
    ys = torch.ones(10, 10, 1)
    inter = TorchLinearInterpolator(ts, ys)
    inter(ts[2])
    inter(1.2)


def test_linear_interpolator():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)

    assert torch.allclose(inter(1.0), ys[0, 0])
    assert torch.allclose(inter(ts[0]), ys[0, 0])
    assert torch.allclose(inter(8.0), ys[0, 7])
    assert torch.allclose(inter(ts[7]), ys[0, 7])
    assert torch.allclose(inter(7.2), torch.tensor(7.2))
    assert torch.allclose(inter(5.4), torch.tensor(5.4))


def test_add_point_t():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    inter.add_point(torch.tensor(11.0), torch.tensor([[1.0]]))


def test_add_point_y_mishape():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    with pytest.raises(
        ValueError,
        match=r"You tried to add a new value that doesn't fit self.ys's shape.",
    ):
        inter.add_point(torch.tensor(11.0), torch.tensor([[2.0, 1.0]]))


@pytest.mark.parametrize("index", [0, 4, 9])
def test_add_point_replaces_existing_timestamp(index):
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    new_t = ts[index]
    inter.add_point(new_t, torch.tensor([[42.0]]))
    torch.testing.assert_close(inter.ts, ts)
    assert torch.all(torch.diff(inter.ts) > 0)
    expected = ys.clone()
    expected[:, index] = 42.0
    torch.testing.assert_close(inter.ys, expected)
    torch.testing.assert_close(inter(new_t), torch.tensor([[42.0]]))


def test_add_point_inserts_middle_in_sorted_order():
    inter = TorchLinearInterpolator(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[[1.0], [2.0], [3.0]]]),
    )
    inter.add_point(torch.tensor(2.5), torch.tensor([[100.0]]))
    torch.testing.assert_close(inter.ts, torch.tensor([1.0, 2.0, 2.5, 3.0]))
    torch.testing.assert_close(inter(torch.tensor(2.25)), torch.tensor([[51.0]]))
    torch.testing.assert_close(inter(torch.tensor(2.75)), torch.tensor([[51.5]]))


def test_add_point_replacement_preserves_gradients():
    original = torch.tensor([[[1.0], [2.0], [3.0]]], requires_grad=True)
    replacement = torch.tensor([[5.0]], requires_grad=True)
    inter = TorchLinearInterpolator(torch.tensor([0.0, 1.0, 2.0]), original)
    inter.add_point(torch.tensor(1.0), replacement)
    inter.ys.sum().backward()
    torch.testing.assert_close(original.grad, torch.tensor([[[1.0], [0.0], [1.0]]]))
    torch.testing.assert_close(replacement.grad, torch.ones_like(replacement))


def test_add_point_end():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    new_t = torch.tensor(11.0)
    inter.add_point(new_t, torch.tensor([[2.0]]))
    assert torch.allclose(inter.ts[-1], new_t)


def test_add_point_start():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    new_t = torch.tensor(-1.0)
    inter.add_point(new_t, torch.tensor([[2.0]]))
    assert torch.allclose(inter.ts[0], new_t)
