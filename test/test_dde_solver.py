import pytest
import torch
from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Dopri5, Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4()])
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
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("solver", [Dopri5()])
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
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]), rtol=1e-5, atol=1e-6)


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
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]), rtol=1e-5, atol=1e-6)
