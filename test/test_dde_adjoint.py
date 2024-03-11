import pytest
import torch
import torch.nn as nn
from torchdde import integrate
from torchdde.solver import Euler, ImplicitEuler, Ralston, RK2, RK4


@pytest.mark.parametrize("solver", [Euler(), RK2(), Ralston(), RK4(), ImplicitEuler()])
def test_learning_delay_in_convex_case(solver):
    class SimpleNDDE(nn.Module):
        def __init__(self, dim, list_delays):
            super().__init__()
            self.in_dim = dim * (1 + len(list_delays))
            self.delays = nn.Parameter(list_delays)
            self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)
            self.init_weight()

        def init_weight(self):
            with torch.no_grad():
                self.linear.weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))

        def forward(self, t, z, args, *, history):
            z__history = z * history[0]
            inp = torch.cat([z, z__history], dim=-1)
            return self.linear(inp)

    def simple_dde(t, y, args, *, history):
        return y * (1 - history[0])

    history_values = torch.tensor([[2.0]])
    history_values = history_values.view(history_values.shape[0], -1)
    history_function = lambda t: history_values

    ts = torch.linspace(0, 10, 101)
    list_delays = torch.tensor([1.0])
    ys = integrate(simple_dde, solver, ts, history_function, None, list_delays)

    learnable_delays = torch.abs(torch.randn((len(list_delays),)))
    model = SimpleNDDE(dim=1, list_delays=learnable_delays)
    lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

    for _ in range(2000):
        model.linear.weight.requires_grad = False
        opt.zero_grad()
        ret = integrate(model, solver, ts, history_function, None, model.delays)
        loss = lossfunc(ret, ys)
        loss.backward()
        opt.step()
        if loss < 1e-6:
            break
    print(model.delays, list_delays)
    assert torch.allclose(model.delays, list_delays, atol=0.01, rtol=0.01)
