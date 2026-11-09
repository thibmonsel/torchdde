import pytest
import torch

from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5, Euler, ImplicitEuler, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), ImplicitEuler(), RK2(), RK4(), Bosh3(), Dopri5()])
def test_reverse_ode(solver):
    ts = torch.linspace(1, 0, 101)
    ys = integrate(
        lambda t, y, args: y, solver, ts[0], ts[-1], ts,
        torch.full((1, 1), torch.e), None, dt0=ts[1] - ts[0],
    )
    assert torch.allclose(ys[0, :, 0], ts.exp(), rtol=0.01, atol=0.01)


def test_reverse_adaptive_ode_selects_negative_initial_step():
    ts = torch.linspace(1, 0, 11)
    ys = integrate(
        lambda t, y, args: y, Dopri5(), ts[0], ts[-1], ts,
        torch.full((1, 1), torch.e), None,
        stepsize_controller=AdaptiveStepSizeController(atol=1e-6, rtol=1e-4),
    )
    assert torch.allclose(ys[0, :, 0], ts.exp(), rtol=0.001, atol=0.001)


def test_reverse_dde_requires_future_data():
    ts = torch.linspace(1, 0, 11)
    with pytest.raises(ValueError, match="future values"):
        integrate(
            lambda t, y, args, history: history[0], Euler(), ts[0], ts[-1], ts,
            lambda t: torch.ones(1, 1), None,
            dt0=ts[1] - ts[0], delays=torch.tensor([0.5]),
        )


@pytest.mark.parametrize("dde", [False, True])
def test_completion_at_step_limit(dde):
    ts = torch.tensor([0.0, 0.5, 1.0])
    if dde:
        func = lambda t, y, args, history: -y
        y0 = lambda t: torch.ones(1, 1)
    else:
        func = lambda t, y, args: -y
        y0 = torch.ones(1, 1)
    kwargs = {"delays": torch.tensor([2.0])} if dde else {}
    ys = integrate(func, Euler(), ts[0], ts[-1], ts, y0, None,
                   dt0=ts[1] - ts[0], max_steps=2, **kwargs)
    assert torch.allclose(ys[:, -1], torch.tensor([[0.25]]))
    with pytest.raises(RuntimeError, match="Maximum number of steps"):
        integrate(func, Euler(), ts[0], ts[-1], ts, y0, None,
                  dt0=ts[1] - ts[0], max_steps=1, **kwargs)


@pytest.mark.parametrize("dde", [False, True])
@pytest.mark.parametrize("module", [False, True])
def test_zero_duration_returns_initial_value_and_gradient(dde, module):
    class Field(torch.nn.Module):
        def forward(self, t, y, args, **kwargs):
            raise AssertionError("A zero-duration solve must not evaluate the field")

    initial = torch.full((2, 3), 3.0, dtype=torch.float64, requires_grad=True)
    y0 = (lambda t: initial) if dde else initial
    func = Field() if module else Field().forward
    ts = torch.zeros(2, dtype=torch.float64)
    kwargs = {"delays": torch.tensor([0.5])} if dde else {}
    ys = integrate(func, Euler(), ts[0], ts[-1], ts, y0, None, **kwargs)
    assert ys.shape == (2, 2, 3)
    assert ys.dtype == initial.dtype
    assert torch.equal(ys[:, 0], initial)
    assert torch.equal(ys[:, 1], initial)
    ys.sum().backward()
    assert torch.equal(initial.grad, torch.full_like(initial, 2.0))


def test_zero_duration_rejects_times_outside_interval():
    with pytest.raises(ValueError, match="requested times"):
        integrate(lambda t, y, args: y, Euler(), torch.tensor(0.0),
                  torch.tensor(0.0), torch.tensor([0.0, 1.0]),
                  torch.ones(1, 1), None)
