# # %%
# %load_ext autoreload
# %autoreload 2
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from dde_solver import *
from interpolators import TorchLinearInterpolator
from model import NDDE, SimpleNDDE, SimpleNDDE2
from nnde_adjoint import nddesolve_adjoint
from ode_solver import *
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import RK23

warnings.filterwarnings("ignore")
seaborn.set_context(context="paper")
seaborn.set_style(style="darkgrid")


def simple_dde(t, y, *, history):
    return  y * (1 - history[0])


def simple_dde2(t, y, *, history):
    return -history[0]

def simple_dde3(t, y, *, history):
    return 0.25 * (history[0]) / (1.0 + history[0] ** 10) - 0.1 * y
    # return 1/2*y -history[0]

device = "cpu"
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_function = lambda t: history_values
print("history_values", history_values.shape)

ts = torch.linspace(0, 10, 101)
list_delays = [1.0]
solver = RK4()
dde_solver = DDESolver(solver, list_delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
print(ys.shape)

for i in range(ys.shape[0]):
    plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close() 

model = NDDE(history_values.shape[-1], list_delays)
try : 
    model.init_weight(1/2)
except:
    pass

model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=10e-4)
losses = []
lens = []

max_epoch = 5000
for i in range(max_epoch):
    opt.zero_grad()
    # history, ts_data, traj = history, ts_history, ys
    # # history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 50 == 0:
        for i in range(ys.shape[0]):
            plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
            plt.plot(ret[i].cpu().detach().numpy(), "--")
        plt.legend()
        plt.pause(2)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}".format(i, loss.item()))

    losses.append(loss.item())
    
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break
