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


class NDDE(nn.Module):
    def __init__(self, dim, list_delays, width=64):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, dim, bias=False)
            # nn.Linear(self.in_dim, width),
            # nn.ReLU(),
            # nn.Linear(width, width),
            # nn.ReLU(),
            # nn.Linear(width, width),
            # nn.ReLU(),
            # nn.Linear(width, dim),
        )

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.model(inp)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 0.0)
            # m.bias.data.fill_(0.01)


# %%
def get_batch(
    ts,
    ys,
    list_delays,
    length=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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


def vf(t, y, history):
    return 0.25 * history[0] / (1.0 + history[0] ** 10) - 0.1 * y


def vf2(t, y, history):
    return -history[0]


def integrate(func, y0, t0, tf, dt, history, delays):
    num_steps = int((tf - t0) / dt)
    values = [y0]
    alltimes = [t0]
    val, t_current = y0, t0
    for i in range(num_steps + 1):
        val = val + dt * func(
            t_current, val, history=[history(t_current - tau) for tau in delays]
        )
        t_current = torch.add(t_current, dt)
        history.add_point(t_current, val)
        values.append(val)
        alltimes.append(t_current)
    alltimes = torch.tensor(alltimes)
    values = torch.hstack(values)
    return alltimes, torch.unsqueeze(values, -1), history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_delays, number_datapoints = [1.0], 256
uniform_dist = torch.distributions.Uniform(0.1, 2.0)
y0_history = uniform_dist.sample(sample_shape=(number_datapoints, 1))
ts_history, ts = torch.tensor([0.0, max(list_delays)]), torch.linspace(1, 10, 250 + 1)
ys = torch.empty((number_datapoints, ts.shape[0], 1))

history = TorchLinearInterpolator(
    ts_history, torch.hstack([y0_history, y0_history])[..., None], y0_history.device
)
_, ys, _ = integrate(
    vf2, y0_history, ts[0], ts[-1], ts[1] - ts[0], history, list_delays
)


dt = ts[1] - ts[0]
length = 3 * int(max(list_delays) / dt)
integration_options = {
    "nSteps": length - 1,
    "dt": dt,
    "t0": ts_history[-1],
    "eval_idx": np.arange(length + 1),
}
model = NDDE(1, list_delays, width=64).to(ys.dtype).cuda()
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-9)
losses = []
lens = []


for i in range(10000):
    opt.zero_grad()
    history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
    ret = nddesolve_adjoint(history, model, integration_options)
    loss = lossfunc(ret.permute(1, 0, 2), traj)
    loss.backward()
    opt.step()
    if i % 100 == 0:
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        plt.plot(traj[0].cpu().detach().numpy())
        plt.plot(ret.permute(1, 0, 2)[0].cpu().detach().numpy(), "--")
        plt.pause(1)
        plt.close()
    print("Epoch : {:4d}, length : {:4d}, Loss : {:.3e}".format(i, length, loss.item()))
    losses.append(loss.item())
    lens.append(length)
    if losses[-1] < 1e-5:
        length += 1
        integration_options = {
            "nSteps": length - 1,
            "dt": dt,
            "t0": ts_history[-1],
            "eval_idx": np.arange(length + 1),
        }

    if length > ys.shape[1] - 10:
        break

# %%
# plt.plot(ret[:, 0].detach().cpu().numpy())
# plt.plot(traj[0, :].detach().cpu())
# plt.show()
# %%
# plt.semilogys
# %%
# history, ts_data, traj = get_batch(ts, ys, list_delays, device=device, length=length)
history, ts_data, traj = get_batch(ts, ys, list_delays, length=length)
inference_options = {
    "nSteps": length - 1,
    "dt": dt,
    "t0": ts_history[-1],
    "eval_idx": np.arange(length + 1),
}
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
