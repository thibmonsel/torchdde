import pytest
import torch
from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Dopri5, Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4(), Dopri5()])
def test_explicit_solver_constant(solver):
    vf = lambda t, y, args: -y
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(ys[:, -1], y0 * torch.exp(-ts[-1]), rtol=10e-3, atol=10e-3)


@pytest.mark.parametrize("solver", [Euler(), RK2()])
def test_explicit_solver_constant2(solver):
    vf = lambda t, y, args: t + t**2
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(
        ys[:, -1], y0 + ts[-1] ** 2 / 2 + ts[-1] ** 3 / 3, rtol=10e-3, atol=10e-3
    )


@pytest.mark.skip(
    reason="RK stages for only time dependent DE don't respect the y0 shape natively"
)
@pytest.mark.parametrize("solver", [Ralston(), RK4()])
def test_explicit_solver_constant3(solver):
    vf = lambda t, y, args: t + t**2
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    ys = integrate(vf, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1] - ts[0])
    assert torch.allclose(
        ys[:, -1], y0 + ts[-1] ** 2 / 2 + ts[-1] ** 3 / 3, rtol=10e-3, atol=10e-3
    )


@pytest.mark.parametrize("solver", [Dopri5()])
def test_explicit_solver_adaptive(solver):
    vf = lambda t, y, args: -y
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    rtol, atol, pcoeff, icoeff, dcoeff = 1e-4, 1e-6, 0.0, 1.0, 0.0
    controller = AdaptiveStepSizeController(
        rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff
    )
    ys = integrate(
        vf, solver, ts[0], ts[-1], ts, y0, None, controller, dt0=ts[1] - ts[0]
    )
    assert torch.allclose(ys[:, -1], y0 * torch.exp(-ts[-1]), rtol=10e-3, atol=10e-3)


@pytest.mark.skip(
    reason="Fix integration for only time dependent \
        functions for adaptive step size controllers"
)
@pytest.mark.parametrize("solver", [Dopri5()])
def test_explicit_solver_adaptive2(solver):
    vf = lambda t, y, args: t + t**2
    ts = torch.linspace(0, 5, 500)
    y0 = torch.rand((10, 1))
    rtol, atol, pcoeff, icoeff, dcoeff = 1e-4, 1e-6, 0.0, 1.0, 0.0
    controller = AdaptiveStepSizeController(
        rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff
    )
    ys = integrate(
        vf, solver, ts[0], ts[-1], ts, y0, None, controller, dt0=ts[1] - ts[0]
    )
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
