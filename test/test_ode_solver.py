import pytest
import torch
from torchdde import integrate
from torchdde.solver import Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4()])
def test_explicit_solver(solver):
    vf = lambda t, y, args: -y
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(ys[:, -1], y0 * torch.exp(-ts[-1]), rtol=10e-3, atol=10e-3)


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4()])
def test_explicit_solver2(solver):
    vf = lambda t, y, args: t + t**2
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(
        ys[:, -1], y0 + ts[-1] ** 2 / 2 + ts[-1] ** 3 / 3, rtol=10e-3, atol=10e-3
    )


@pytest.mark.parametrize("solver", [ImplicitEuler()])
def test_implicit_solver(solver):
    vf = lambda t, y, args: -y
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(ys[:, -1], y0 * torch.exp(-ts[-1]), rtol=10e-3, atol=10e-3)


@pytest.mark.parametrize("solver", [ImplicitEuler()])
def test_implicit_solver2(solver):
    vf = lambda t, y, args: t + t**2
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(
        ys[:, -1], y0 + ts[-1] ** 2 / 2 + ts[-1] ** 3 / 3, rtol=10e-3, atol=10e-3
    )
