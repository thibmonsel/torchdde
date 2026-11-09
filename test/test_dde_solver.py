import pytest
import torch
from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5, Euler, ImplicitEuler, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), RK4(), Bosh3(), Dopri5()])
def test_multiple_delay_gradients_match_finite_difference(solver):
    ts = torch.linspace(0.5, 1.5, 21, dtype=torch.float64)
    delays = torch.tensor([0.21, 0.37], dtype=ts.dtype, requires_grad=True)

    def terminal_value(lags):
        ys = integrate(
            lambda t, y, args, history: 0.1 * y + 0.2 * history[0] - 0.1 * history[1],
            solver,
            ts[0],
            ts[-1],
            ts,
            lambda t: torch.exp(t).reshape(1, 1),
            None,
            delays=lags,
            dt0=ts[1] - ts[0],
            discretize_then_optimize=True,
        )
        return ys[:, -1].sum()

    (gradient,) = torch.autograd.grad(terminal_value(delays), delays)
    eps = 1e-5
    expected = []
    for offset in torch.eye(2, dtype=ts.dtype) * eps:
        plus = terminal_value(delays.detach() + offset)
        minus = terminal_value(delays.detach() - offset)
        expected.append((plus - minus) / (2 * eps))
    torch.testing.assert_close(gradient, torch.stack(expected), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("solver", [Euler(), RK2(), RK4(), Bosh3(), Dopri5()])
def test_delay_gradient_at_first_step_history_boundary(solver):
    # A stage at t=tau must retain the full history derivative, even at t-tau=0.
    ts = torch.tensor([0.0, 0.125], dtype=torch.float64)
    delay = torch.tensor([0.125], dtype=ts.dtype, requires_grad=True)
    ys = integrate(
        lambda t, y, args, history: history[0],
        solver,
        ts[0],
        ts[-1],
        ts,
        lambda t: (1 + t).reshape(1, 1),
        None,
        delays=delay,
        dt0=ts[-1],
        discretize_then_optimize=True,
    )
    (gradient,) = torch.autograd.grad(ys[:, -1].sum(), delay)
    # Include the step-size cap's derivative when delay == dt. Higher-order
    # methods integrate the linear-in-time derivative exactly; Euler uses t=0.
    capped_dt = torch.clamp(ts[-1], max=delay.min())
    reference = 1 + capped_dt * (1 - delay)
    if not isinstance(solver, Euler):
        reference = reference + capped_dt.square() / 2
    (expected_gradient,) = torch.autograd.grad(reference.sum(), delay)
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("solver", [Euler(), RK2(), RK4()])
def test_explicit_solver_constant(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    ys = integrate(
        vf,
        solver,
        ts[0],
        ts[-1],
        ts,
        lambda t: y0,
        None,
        delays=torch.tensor([1.0]),
        discretize_then_optimize=True,
        dt0=ts[1] - ts[0],
    )
    # for t in [0,1] solution is y0 * (t -1)
    assert torch.allclose(
        ys[:, :80, 0].flatten(), (y0 * (1 - ts[:80])).flatten(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("solver", [Dopri5(), Bosh3()])
def test_explicit_solver_adaptive(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    rtol, atol, pcoeff, icoeff, dcoeff = 1e-3, 1e-6, 0.0, 1.0, 0.0
    controller = AdaptiveStepSizeController(
        rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff
    )
    ys = integrate(
        vf,
        solver,
        ts[0],
        ts[-1],
        ts,
        lambda t: y0,
        None,
        stepsize_controller=controller,
        delays=torch.tensor([1.0]),
        discretize_then_optimize=True,
        dt0=ts[1] - ts[0],
    )
    # for t in [0,1] solution is y0 * (t -1)
    assert torch.allclose(
        ys[:, :80, 0].flatten(), (y0 * (1 - ts[:80])).flatten(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("solver", [ImplicitEuler()])
def test_implicit_solver(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    ys = integrate(
        vf,
        solver,
        ts[0],
        ts[-1],
        ts,
        lambda t: y0,
        None,
        delays=torch.tensor([1.0]),
        discretize_then_optimize=True,
        dt0=ts[1] - ts[0],
    )
    assert torch.allclose(
        ys[:, :80, 0].flatten(), (y0 * (1 - ts[:80])).flatten(), rtol=1e-5, atol=1e-6
    )
