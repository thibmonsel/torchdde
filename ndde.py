import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (
    RK2,
    RK4,
    DDESolver,
    Euler,
    Ralston,
    TorchLinearInterpolator,
    nddesolve_adjoint,
)

warnings.filterwarnings("ignore")
seaborn.set_context(context="paper")
seaborn.set_style(style="darkgrid")


def simple_dde(t, y, *, history):
    return y * (1 - history[0])


def simple_dde2(t, y, *, history):
    return -history[0] - history[1]


def simple_dde3(t, y, *, history):
    return 0.25 * (history[0]) / (1.0 + history[0] ** 10) - 0.1 * y
    # return 1/2*y -history[0]


device = "cpu"
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_function = lambda t: history_values
print("history_values", history_values.shape)

ts = torch.linspace(0, 10, 101)
list_delays = [0.5, 1.0]
solver = RK4()
dde_solver = DDESolver(solver, list_delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
print(ys.shape)

for i in range(ys.shape[0]):
    plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close()

model = NDDE(history_values.shape[-1], list_delays)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)
losses = []
lens = []

mask = np.logspace(1, 1e-1, ts.shape[0]) / 10
mask = torch.tensor(mask.reshape(1, mask.shape[0], 1), device=device)

max_epoch = 5000
for i in range(max_epoch):
    opt.zero_grad()
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 50 == 0:
        for i in range(ys.shape[0]):
            plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
            plt.plot(ret[i].cpu().detach().numpy(), "--")
        plt.legend()
        plt.savefig("last_res.png", bbox_inches="tight", dpi=100)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}".format(i, loss.item()))

    losses.append(loss.item())

    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break
