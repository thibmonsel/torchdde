# # %%
# %load_ext autoreload
# %autoreload 2
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from interpolators import TorchLinearInterpolator
from nnde_adjoint import nddesolve_adjoint
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import RK23


warnings.filterwarnings("ignore")
seaborn.set_context(context="paper")
seaborn.set_style(style="darkgrid")


class NDDE(nn.Module):
    def __init__(self, dim, list_delays, width=64):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.mlp = nn.Sequential(nn.Linear(self.in_dim,32),
          nn.ReLU(),
          nn.Linear(32,32),
          nn.ReLU(),
          nn.Linear(32,dim))

        # self.Params = torch.nn.parameter.Parameter(
        #     1.65 * torch.ones((2,), dtype=torch.float32, requires_grad=True)
        # )

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.mlp(inp)
        # return self.Params[0] * z * (1.0 - self.Params[1] * history[0])


def get_batch(
    ts,
    ys,
    list_delays,
    device="cpu",  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    ts : [N_t]
    ys : [B, N_t, #features]
    """
    dt = ts[1] - ts[0]
    max_delay = max(list_delays)
    max_delay_idx = int(max_delay / dt)
    # pick random indices for each batch
    rand_idx = np.random.choice(ys.shape[1] - max_delay_idx - length - 1)
    # history_batch : [batch_size, max_delay_idx, #features]
    # ts_history : [length] negative time
    # data_batch : [batch_size, length, #features]
    history_batch = ys[:, rand_idx : rand_idx + max_delay_idx + 1]
    ts_history = torch.linspace(0, max_delay, max_delay_idx + 1)
    data_batch = ys[:, rand_idx + max_delay_idx : rand_idx + max_delay_idx + length + 1]
    ts_data = torch.linspace(max_delay, float(length * dt), length + 1)
    interpolator = TorchLinearInterpolator(ts_history, history_batch, device)
    data_batch = data_batch.to(device)
    return interpolator, ts_data, data_batch


def vf2(t, y, history):
    return y * (1.0 - history[0])


def integrate(func, y0, ts, history, delays):
    values = [y0]
    alltimes = [ts[0]]
    val, t_current, dt = y0, ts[0], ts[1] - ts[0]
    for t_current in ts[1:]:
        # euler
        # val = val + dt * func(
        #     t_current, val, history=[history(t_current - tau) for tau in delays]
        # )
        # rk2
        k1 = func(t_current, val, history=[history(t_current - tau) for tau in delays])
        k2 = func(
            t_current + dt,
            val + dt * k1,
            history=[
                history(t_current + dt - tau) if t_current + dt - tau > ts[0] else y0
                for tau in delays
            ],
        )
        val = val + dt / 2 * (k1 + k2)
        history.add_point(t_current, val)
        values.append(val)
        alltimes.append(t_current)

    alltimes = torch.tensor(alltimes)
    values = torch.hstack(values)
    return alltimes, torch.unsqueeze(values, -1), history


device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_delays, number_datapoints = [1.0], 1
y0_history = torch.tensor([2.0]).reshape(number_datapoints, 1)
ts_history, ts = torch.tensor([-max(list_delays), 0.0]), torch.linspace(0, 10, 100 + 1)
ys = torch.empty((number_datapoints, ts.shape[0], 1))
ts_history = ts_history.to(device)
ys = ys.to(device)
y0_history = y0_history.to(device)

# history = TorchLinearInterpolator(
#     ts_history, torch.hstack([y0_history, y0_history])[..., None], y0_history.device
# )

history = TorchLinearInterpolator(
    ts_history, torch.hstack([y0_history, y0_history])[..., None])
_, ys, _ = integrate(vf2, y0_history, ts, history, list_delays)


model = NDDE(1, list_delays, width=64).to(ys.dtype)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
losses = []
lens = []

print(ys.shape, ts.shape)
for i in range(5000):
    opt.zero_grad()
    history, ts_data, traj = history, ts_history, ys
    # history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
    ret = nddesolve_adjoint(history, model, ts)
    loss = lossfunc(ret.permute(1, 0, 2), traj)
    loss.backward()
    opt.step()
    if i % 100 == 0:

        plt.plot(traj[0].cpu().detach().numpy(), label="Truth")
        plt.plot(ret.permute(1, 0, 2)[0].cpu().detach().numpy(), "--")
        plt.legend()
        plt.pause(2)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}".format(i, loss.item()))

    losses.append(loss.item())
    if losses[-1] < 1e-5:
        plt.plot(traj[0].cpu().detach().numpy())
        plt.plot(ret.permute(1, 0, 2)[0].cpu().detach().numpy(), "--")
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
