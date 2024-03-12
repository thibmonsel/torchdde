import pytest
import torch
from torchdde import integrate
from torchdde.solver import Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4()])
def test_explicit_solver(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    ys = integrate(
        vf,
        solver,
        ts,
        lambda t: y0,
        None,
        delays=torch.tensor([1.0]),
        discretize_then_optimize=True,
    )
    # for t in [0,1] solution is y0 * (t -1)
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]))


@pytest.mark.parametrize("solver", [ImplicitEuler()])
def test_implicit_solver(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    ys = integrate(
        vf,
        solver,
        ts,
        lambda t: y0,
        None,
        delays=torch.tensor([1.0]),
        discretize_then_optimize=True,
    )
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]))
