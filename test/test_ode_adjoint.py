import pytest
import torch
import torch.nn as nn
from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Dopri5, Euler, ImplicitEuler, Ralston, RK2, RK4


class SimpleNODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Linear(1, 1, bias=False)
        self.init_weight()

    def init_weight(self):
        with torch.no_grad():
            self.params.weight = nn.Parameter(-torch.tensor([1.0]))

    def forward(self, t, z, args):
        return self.params.weight * z


def simple_ode(t, y, args):
    return -2.0 * y


ts = torch.linspace(0, 10, 101)
y0 = torch.rand((2, 3))
with torch.no_grad():
    ys = integrate(
        simple_ode,
        RK2(),
        ts[0],
        ts[-1],
        ts,
        y0,
        None,
        dt0=ts[1] - ts[0],
        discretize_then_optimize=True,
    )

model = SimpleNODE()
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)

for _ in range(2000):
    opt.zero_grad()
    ret = integrate(model, RK2(), ts[0], ts[-1], ts, y0, None)
    loss = lossfunc(ret, ys)

    loss.backward()
    opt.step()
    if loss < 10e-8:
        break

assert torch.allclose(ys, ret, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4(), ImplicitEuler()])
def test_very_simple_system(solver):
    class SimpleNODE(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.Linear(1, 1, bias=False)
            self.init_weight()

        def init_weight(self):
            with torch.no_grad():
                self.params.weight = nn.Parameter(-torch.tensor([1.0]))

        def forward(self, t, z, args):
            return self.params.weight * z

    def simple_ode(t, y, args):
        return -2.0 * y

    ts = torch.linspace(0, 10, 101)
    y0 = torch.rand((2, 3))
    with torch.no_grad():
        ys = integrate(
            simple_ode,
            solver,
            ts[0],
            ts[-1],
            ts,
            y0,
            None,
            dt0=ts[1] - ts[0],
            discretize_then_optimize=True,
        )

    model = SimpleNODE()
    lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)

    for _ in range(2000):
        opt.zero_grad()
        ret = integrate(model, solver, ts[0], ts[-1], ts, y0, None)
        loss = lossfunc(ret, ys)

        loss.backward()
        opt.step()
        if loss < 10e-8:
            break

    assert torch.allclose(ys, ret, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("solver", [Dopri5()])
def test_very_simple_system2(solver):
    class SimpleNODE(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.Linear(1, 1, bias=False)
            self.init_weight()

        def init_weight(self):
            with torch.no_grad():
                self.params.weight = nn.Parameter(-torch.tensor([1.0]))

        def forward(self, t, z, args):
            return self.params.weight * z

    def simple_ode(t, y, args):
        return -2.0 * y

    ts = torch.linspace(0, 10, 101)
    y0 = torch.rand((2, 3))
    with torch.no_grad():
        ys = integrate(
            simple_ode,
            solver,
            ts[0],
            ts[-1],
            ts,
            y0,
            None,
            discretize_then_optimize=True,
        )

    model = SimpleNODE()
    lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)

    for _ in range(2000):
        opt.zero_grad()
        stepsize_controller = AdaptiveStepSizeController(1e-3, 1e-6)
        ret = integrate(
            model,
            solver,
            ts[0],
            ts[-1],
            ts,
            y0,
            None,
            stepsize_controller=stepsize_controller,
        )
        loss = lossfunc(ret, ys)

        loss.backward()
        opt.step()
        if loss < 10e-8:
            break

    assert torch.allclose(ys, ret, atol=0.01, rtol=0.01)
