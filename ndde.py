import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, ddesolve_adjoint)

warnings.filterwarnings("ignore")
seaborn.set_context(context="paper")
seaborn.set_style(style="darkgrid")

def simple_dde(t, y, *, history):
    return y * (1 - history[0])


def simple_dde2(t, y, *, history):
    return 1/2 * history[0] - history[1]


def simple_dde3(t, y, *, history):
    return 0.25 * (history[0]) / (1.0 + history[0] ** 10) - 0.1 * y
    # return 1/2*y -history[0]

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_interpolator = TorchLinearInterpolator(
    torch.tensor(([-10.0, 0.0])),
    torch.concat((history_values, history_values), dim=1)[...,None],
)
history_interpolator.to(device)
history_function = lambda t: history_interpolator(t)
print("history_values", history_values.shape)

ts = torch.linspace(0, 20, 201)
ts = ts.to(device)
list_delays = torch.tensor([1.0, 2.0])
list_delays = list_delays.to(device)    
solver = RK4()
dde_solver = DDESolver(solver, list_delays)
ys, _ = dde_solver.integrate(simple_dde2, ts, history_function)
print(ys.shape)

for i in range(ys.shape[0]):
    plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close()

# 2 delays for brusselator looks like a good choice
learnable_delays = torch.abs(torch.randn((len(list_delays),)))
learnable_delays = learnable_delays.to(device)
model = NDDE(history_values.shape[-1], learnable_delays, width=32)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
losses = []
lens = []
max_epoch = 10000
for i in range(max_epoch):
    opt.zero_grad()
    t = time.time()
    ret, _ = dde_solver.integrate(model, ts, history_function)
    # ret = ddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 15 == 0:
        for i in range(ys.shape[0]):
            plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
            plt.plot(ret[i].cpu().detach().numpy(), "--")
        plt.legend()
        plt.savefig("last_res.png", bbox_inches="tight", dpi=100)
        plt.close()
    print(
        "Epoch : {:4d}, Loss : {:.3e}, Time {}, tau : {} & {}".format(
            i, loss.item(), time.time()-t, model.delays[0],  model.delays[1]
        )
    )

    losses.append(loss.item())
    idx = np.random.randint(0, ys.shape[0])
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[idx].cpu().detach().numpy())
        plt.plot(ret[idx].cpu().detach().numpy(), "--")
        plt.show()
        break
