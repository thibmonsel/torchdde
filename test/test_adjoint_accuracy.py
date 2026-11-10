"""Accuracy of adjoint gradients against exact references.

The delay (0.2) is a multiple of the output step (0.05), so every breakpoint
of the adjoint dynamics lies on an output time. In that case DTO parameter
gradients agree with finite differences to ~1e-9 and serve as the reference.
"""
import pytest
import torch
from torch import nn

from torchdde import Dopri5, integrate
from torchdde.global_interpolation.linear_interpolation import TorchLinearInterpolator


def test_linear_interpolator_one_sided_jump_values():
    ts = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    ys = torch.tensor([[[0.0], [10.0], [20.0]]], dtype=torch.float64)
    interp = TorchLinearInterpolator(ts, ys)
    # Jump at t=1: 10 seen from below (and at 1), 4 just above.
    interp.add_point(ts[1], ys[:, 1], right_y=torch.tensor([[4.0]], dtype=torch.float64))
    at = lambda t: interp(torch.tensor(t, dtype=torch.float64)).item()
    assert at(0.5) == pytest.approx(5.0)
    assert at(1.0) == pytest.approx(10.0)
    assert at(1.5) == pytest.approx(12.0)  # halfway from 4 to 20
    many = interp.evaluate_many(torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64))
    torch.testing.assert_close(many[:, 0, 0], torch.tensor([5.0, 10.0, 12.0], dtype=torch.float64))
    # A later continuous point keeps the jump intact.
    interp.add_point(torch.tensor(1.5, dtype=torch.float64), torch.tensor([[12.0]], dtype=torch.float64))
    assert at(1.25) == pytest.approx(8.0)


def test_linear_interpolator_without_jumps_is_unchanged():
    ts = torch.tensor([0.0, 1.0, 2.0])
    ys = torch.tensor([[[0.0], [10.0], [30.0]]])
    interp = TorchLinearInterpolator(ts, ys)
    interp.add_point(torch.tensor(1.5), torch.tensor([[15.0]]))
    assert interp.ys_right is None
    assert interp(torch.tensor(1.25)).item() == pytest.approx(12.5)


def _problem(loss_mode):
    torch.manual_seed(0)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.delays = nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
            self.net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).double()

        def forward(self, t, y, args, *, history):
            return self.net(torch.cat([y, *history], dim=-1))

    model = Model()
    ts = torch.linspace(0, 1, 21, dtype=torch.float64)
    y0 = torch.rand(4, 2, dtype=torch.float64) + 1

    def gradients(dto):
        model.zero_grad()
        ys = integrate(
            model, Dopri5(), ts[0], ts[-1], ts, lambda t: y0, None,
            dt0=ts[1] - ts[0], delays=model.delays, discretize_then_optimize=dto,
        )
        ys = ys[:, -1:] if loss_mode == "final" else ys
        (ys - y0[:, None]).square().mean().backward()
        return [p.grad.clone() for p in model.parameters()]

    return gradients


def _relative_error(actual, expected):
    return max(((a - e).norm() / e.norm()).item() for a, e in zip(actual, expected))


def test_adjoint_parameter_gradients_with_loss_jumps():
    gradients = _problem("mean")
    # The mean loss makes the adjoint jump at every output time. Before the
    # adjoint history stored both one-sided values this error was 2.5e-3.
    assert _relative_error(gradients(False)[1:], gradients(True)[1:]) < 1e-3
