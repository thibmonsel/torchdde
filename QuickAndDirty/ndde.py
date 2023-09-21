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
from model import NDDE, SimpleNDDE
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


device = "cpu"
history_values = torch.tensor([3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_function = lambda t: history_values
print("history_values", history_values.shape)

ts = torch.linspace(0, 10, 501)
list_delays = [1.0]
solver = RK4()
dde_solver = DDESolver(solver, list_delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
print(ys.shape)

model = SimpleNDDE(history_values.shape[-1], list_delays)
# try : 
#     model.init_weight(1.75)
# except:
#     pass
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
losses = []
lens = []

for i in range(5000):
    opt.zero_grad()
    # history, ts_data, traj = history, ts_history, ys
    # # history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 50 == 0:
        plt.plot(ys[0].cpu().detach().numpy(), label="Truth")
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.legend()
        plt.pause(2)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}".format(i, loss.item()))

    losses.append(loss.item())
    
    if losses[-1] < 1e-5:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break


# %%
# plt.plot(ret[:, 0].detach().cpu().numpy())
# plt.plot(traj[0, :].detach().cpu())
# plt.show()
# %%
# plt.semilogys
# %%
# history, ts_data, traj = get_batch(ts, ys, list_delays, device=device, length=length)
# history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
# inference_options = {
#     "nSteps": length ,
#     "dt": dt,
#     "t0": ts_history[-1],
#     "eval_idx": np.arange(length + 1),
# }
# ret = nddesolve_adjoint(history, model, inference_options)

# # %%
# fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
# plt.sca(axs[0])
# plt.plot(ret[:, 0].detach().cpu().numpy(), label="Prediction", lw=3)
# plt.plot(traj[0, :].detach().cpu(), "--", label="Truth", lw=3)
# plt.legend()
# plt.sca(axs[1])
# plt.semilogy(losses, lw=2, label="Losses")
# plt.legend(loc="upper center")
# ax = axs[1].twinx()
# ax.plot(lens, label="Traj Length", color="tab:green")
# plt.legend(loc="lower center")
# plt.show()
