"""Accuracy of adjoint gradients against exact references.

The delay (0.2) is a multiple of the output step (0.05), so every breakpoint
of the adjoint dynamics lies on an output time. In that case DTO parameter
gradients agree with finite differences to ~1e-9 and serve as the reference.
"""
import pytest
import torch
from torch import nn

from torchdde import Dopri5, integrate
from torchdde.adjoint_dde import _split_interval
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_split_interval_merges_close_breakpoints(dtype):
    start, end = torch.tensor(0.55, dtype=dtype), torch.tensor(0.5, dtype=dtype)
    breakpoints = [0.4, 0.5 + 1e-12, 0.52, 0.52 + 1e-12, 0.53, 0.55 - 1e-12, 0.6]
    nodes = [float(node) for node in _split_interval(start, end, breakpoints)]
    assert nodes == pytest.approx([0.55, 0.53, 0.52, 0.5])
    assert all(a > b for a, b in zip(nodes, nodes[1:]))
    assert len(_split_interval(start, end, [])) == 2


def _problem(loss_mode, delays=(0.2,), split=True):
    torch.manual_seed(0)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.delays = nn.Parameter(torch.tensor(delays, dtype=torch.float64))
            self.net = nn.Sequential(
                nn.Linear(2 * (1 + len(delays)), 8), nn.Tanh(), nn.Linear(8, 2)
            ).double()

        def forward(self, t, y, args, *, history):
            return self.net(torch.cat([y, *history], dim=-1))

    model = Model()
    ts = torch.linspace(0, 1, 21, dtype=torch.float64)
    y0 = torch.rand(4, 2, dtype=torch.float64) + 1

    def loss(dto, dt0):
        ys = integrate(
            model, Dopri5(), ts[0], ts[-1], ts, lambda t: y0, None,
            dt0=torch.tensor(dt0, dtype=torch.float64), delays=model.delays,
            discretize_then_optimize=dto, max_steps=4096,
            adjoint_split_at_breakpoints=split,
        )
        ys = ys[:, -1:] if loss_mode == "final" else ys
        return (ys - y0[:, None]).square().mean()

    def gradients(dto):
        model.zero_grad()
        loss(dto, 0.05).backward()
        return [p.grad.clone() for p in model.parameters()]

    def delay_finite_difference(eps=1e-6, dt0=0.05 / 64):
        # DTO delay gradients carry their own discretisation error (moving the
        # delay moves breakpoints against a fixed grid), so use a fine forward.
        with torch.no_grad():
            model.delays[0] += eps
            plus = loss(True, dt0).item()
            model.delays[0] -= 2 * eps
            minus = loss(True, dt0).item()
            model.delays[0] += eps
        return (plus - minus) / (2 * eps)

    return gradients, delay_finite_difference


def _relative_error(actual, expected):
    return max(((a - e).norm() / e.norm()).item() for a, e in zip(actual, expected))


@pytest.mark.parametrize(
    "loss_mode, delays",
    [("mean", (0.2,)), ("final", (0.2,)), ("mean", (0.2, 0.4))],
)
def test_adjoint_parameter_gradients_match_exact_reference(loss_mode, delays):
    # Errors were 2.5e-3 / 5.6e-4 / 3.8e-3 before the adjoint stored both
    # one-sided values at loss jumps and read breakpoints from inside each
    # backward interval; they are now ~2e-7.
    gradients, _ = _problem(loss_mode, delays)
    assert _relative_error(gradients(False)[1:], gradients(True)[1:]) < 1e-5


@pytest.mark.parametrize("loss_mode", ["mean", "final"])
def test_adjoint_delay_gradient_matches_finite_difference(loss_mode):
    # Was 6.8e-3 for both losses; now ~1e-4, limited by the reference.
    gradients, delay_finite_difference = _problem(loss_mode)
    reference = delay_finite_difference()
    assert abs(gradients(False)[0].item() - reference) < 1e-3 * abs(reference)


@pytest.mark.parametrize(
    "loss_mode, delays",
    [("mean", (0.23,)), ("final", (0.37,)), ("mean", (0.23, 0.37))],
)
def test_adjoint_gradients_with_delays_off_the_output_grid(loss_mode, delays):
    # Off the grid, breakpoints t_k - tau, T - tau and t0 + tau fall inside
    # output intervals; the backward pass integrates up to each of them.
    # Before, parameter errors were 3e-4 to 7e-4 and delay errors up to 1.5e-2.
    gradients, delay_finite_difference = _problem(loss_mode, delays)
    adjoint, dto = gradients(False), gradients(True)
    # DTO itself is only accurate to ~4e-5 here (its forward crosses breakpoints).
    assert _relative_error(adjoint[1:], dto[1:]) < 1e-4
    if len(delays) == 1:
        reference = delay_finite_difference()
        assert abs(adjoint[0].item() - reference) < 1e-3 * abs(reference)


def test_adjoint_breakpoint_splitting_can_be_disabled():
    # Off the grid, disabling the splitting brings back the cheaper but
    # first-order backward pass; on the grid it changes nothing.
    off_split, _ = _problem("mean", (0.23,), split=True)
    off_plain, _ = _problem("mean", (0.23,), split=False)
    assert _relative_error(off_split(False)[1:], off_plain(False)[1:]) > 1e-5
    on_split, _ = _problem("mean", (0.2,), split=True)
    on_plain, _ = _problem("mean", (0.2,), split=False)
    for a, b in zip(on_split(False), on_plain(False)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
