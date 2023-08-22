# # %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cd_rom_public import cd_rom
from interpolators import TorchLinearInterpolator
from QuickAndDirty.nnde_adjoint import nddesolve_adjoint


# %%
# tmax = 100
# dt = 1e-2
# ts = np.linspace(0, tmax, int(tmax / dt) + 1)
# np_data = np.sin(ts)  # +np.sin(1.27*time)
# print(ys.shape)
# plt.plot(ys)
# plt.show()

# %%
class NDDE(nn.Module):
    def __init__(self, dim, list_delays, width=64):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, dim),
        )

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.model(inp)

# %%
def get_batch(
    ts,
    ys,
    list_delays,
    batch_size=256,
    length=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    ts : [N_t]
    ys : [N_t, #features]
    """
    dt = ts[1] - ts[0]
    max_delay = max(list_delays)
    max_delay_idx = int(max_delay / dt)
    # pick random indices for each batch
    rand_idx = np.random.choice(
        ys.shape[0] - max_delay_idx - length - 1, size=batch_size
    )
    # history_batch : [batch_size, max_delay_idx, #features]
    # ts_history : [length] negative time
    # data_batch : [batch_size, length, #features]
    history_batch = torch.stack([ys[i : i + max_delay_idx + 1] for i in rand_idx])
    ts_history = torch.linspace(-max_delay, 0, max_delay_idx + 1)
    data_batch = torch.stack(
        [ys[i + max_delay_idx : i + max_delay_idx + length + 1] for i in rand_idx]
    )
    ts_data = torch.linspace(0, float(length * dt), length + 1)
    interpolator = TorchLinearInterpolator(ts_history, history_batch, device)
    data_batch = data_batch.to(device)
    return interpolator, ts_data, data_batch


def vf(t, y, history):
    return 0.25 * history[0] / (1.0 + history[0] ** 10) - 0.1 * y

def integrate(func, y0, t0, tf, dt, history, delays):
    num_steps = int((tf-t0)/dt)
    values = [y0]
    alltimes = [t0]
    val, t_current = y0 , t0
    for i in range(num_steps) : 
        val = val + dt * func(t_current, val, history=[history(t_current - tau) for tau in delays])
        t_current = torch.add(t_current, dt)
        history.add_point(t_current, val)
        values.append(val)
        alltimes.append(t_current)
    alltimes = torch.tensor(alltimes)
    values = torch.tensor(values)
    return alltimes, torch.unsqueeze(values, -1) , history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_delays = [10]
dt, y0 = torch.tensor(1e-2), torch.tensor([1.2])
ts_history = torch.tensor([0.0, 10.0]).reshape(2)
ts_history = ts_history.to(device)
y0 = y0.to(device)
history = TorchLinearInterpolator(ts_history, y0.repeat(1, 2, 1), y0.device)
t0, tf = torch.tensor(max(list_delays)), torch.tensor(200.0)
t0, tf = t0.to(device), tf.to(device)

ts, ys, histo = integrate(vf, y0, t0 , tf, dt, history, list_delays)
print(ts.shape, ys.shape)

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ys = torch.tensor(np_data).reshape(-1, 1)
# ts = np.load("ts.npy")
# ys = np.load("ys.npy")
# ts = torch.tensor(ts)
# ys = torch.tensor(ys).reshape(-1, 1)
# dt = ts[1] - ts[0]
# ts, ys = ts.to(device), ys.to(device)
# interpolator, ts_data, data_batch = get_batch(ts, ys, [1.0, 2.0])

# %%
# plt.plot(ys.cpu())
# plt.show()

# %%
length = 800
integration_options = {
    "nSteps": length - 1,
    "dt": dt,
    "t0": 0,
    "eval_idx": np.arange(length),
}
delay_pts = [10]
list_delays = [d  for d in delay_pts]
model = NDDE(1, list_delays, width=64).to(ys.dtype).cuda()
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
losses = []
lens = []


# %%
history, ts_data, traj = get_batch(ts, ys, list_delays, device=device, length=length)
# traj = traj.to(device)
# %%
ret = nddesolve_adjoint(history, model, integration_options)
ret.shape

# %%
for i in range(2000):
    opt.zero_grad()
    history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
    ret = nddesolve_adjoint(history, model, integration_options)
    # loss = torch.mean(torch.sum((ret.permute(1,0,2)-traj)**4,axis=-1)**0.25)
    loss = lossfunc(ret.permute(1, 0, 2), traj[:, 1:])
    loss.backward()
    opt.step()
    print("Epoch : {:4d}, length : {:4d}, Loss : {:.3e}".format(i, length, loss.item()))
    losses.append(loss.item())
    lens.append(length)
    if losses[-1] < 1e-2:
        length += 1
        integration_options = {
            "nSteps": length - 1,
            "dt": dt,
            "t0": 0,
            "eval_idx": np.arange(length),
        }
        
    if length > 2500:
        break

# %%
# plt.plot(ret[:, 0].detach().cpu().numpy())
# plt.plot(traj[0, :].detach().cpu())
# plt.show()
# %%
# plt.semilogys
# %%
# history, ts_data, traj = get_batch(ts, ys, list_delays, device=device, length=length)
inference_options = {"nSteps": 1000, "dt": dt, "t0": 0, "eval_idx": np.arange(1000)}
ret = nddesolve_adjoint(history, model, inference_options)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
plt.sca(axs[0])
plt.plot(ret[:, 0].detach().cpu().numpy(), label="Prediction", lw=3)
plt.plot(traj[0, :].detach().cpu(), "--", label="Truth", lw=3)
plt.legend()
plt.sca(axs[1])
plt.semilogy(losses, lw=2, label="Losses")
plt.legend(loc="upper center")
ax = axs[1].twinx()
ax.plot(lens, label="Traj Length", color="tab:green")
plt.legend(loc="lower center")
plt.show()
