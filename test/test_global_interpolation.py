import pytest
import torch
from torchdde import TorchLinearInterpolator
from torchdde.global_interpolation.dense_interpolation import DenseInterpolation
from torchdde.local_interpolation import FirstOrderPolynomialInterpolation


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


def test_add_point_warning():
    ts = torch.linspace(1.0, 10.0, 10)
    ys = torch.arange(1, 11, dtype=torch.float32)[None, ..., None]
    inter = TorchLinearInterpolator(ts, ys)
    new_t = ts[0]
    with pytest.warns(
        UserWarning,
        match=f"already have new_t={new_t} point in interpolation, overwriting it",
    ):
        inter.add_point(new_t, torch.tensor([[2.0]]))


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
