import pytest
import torch
from torchdde import TorchLinearInterpolator


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
