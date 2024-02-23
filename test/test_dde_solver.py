import pytest
import torch
from torchdde.solver.dde_solver import DDESolver
from torchdde.solver.ode_solver import Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4()])
def test_explicit_solver(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    dde_solver = DDESolver(solver, torch.tensor([1.0]))
    ys, _ = dde_solver.integrate(vf, ts, lambda t: y0, None)
    print(ys.shape, y0.shape, ts.shape, (y0 * ts[:80]).shape)
    # for t in [0,1] solution is y0 * (t -1)
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]))


@pytest.mark.parametrize("solver", [ImplicitEuler()])
def test_implicit_solver(solver):
    vf = lambda t, y, args, history: -history[0]
    ts = torch.linspace(0, 2, 200)
    y0 = torch.rand((10, 1))
    dde_solver = DDESolver(solver, torch.tensor([1.0]))
    ys, _ = dde_solver.integrate(vf, ts, lambda t: y0, None)
    assert torch.allclose(ys[:, :80, 0], y0 * (1 - ts[:80]))
