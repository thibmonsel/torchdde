import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, ddesolve_adjoint)


class SimpleNDDE2(nn.Module):
    def __init__(self, dim, list_delays):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays =  nn.Parameter(list_delays)
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)
        self.init_weight()
        
    def init_weight(self):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[0, 1/2, -1.0]]))

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.linear(inp)


def simple_dde2(t, y, *, history):
    return 1/2 * history[0] - history[1]


device = "cpu"
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], -1)
history_function = lambda t: history_values 
print("history_values", history_values.shape)

ts = torch.linspace(0, 20, 201)
list_delays = [1.0, 2.0]
dde_solver = DDESolver(Euler(), list_delays)
ys, _ = dde_solver.integrate(simple_dde2, ts, history_function)

""" 
First experiment is with two delays and we try to see if the adjoint method is correct for the learning of delays.
We setourselves in the case where we know the vector field and the history function is constant. Therefore,
we have x'(t) = 0 for t < 0 (this nullifies a part of the loss wtr to the delay parameter).

We note the user that the adjoint method is needs to have the same solver for the forward and backward pass in order to have a good optimization.
Else there might be some discrepencies between the optimal delay and the one learned by the adjoint method.
"""

# x, y = np.meshgrid(np.linspace(0.2, 2.0, 20),np.linspace(0.2, 2.0, 20))
# possible_delays = torch.from_numpy(np.concatenate([x.reshape(1,-1),y.reshape(1,-1)],axis=0).T)
# loss_list = []
# for delay in possible_delays:
#     dde_solver = DDESolver(Euler(), delay)
#     ys_other, _ = dde_solver.integrate(simple_dde2, ts, history_function)
#     loss = torch.mean((ys -ys_other)**2)
#     loss_list.append(loss.item())

# loss_list = torch.log(torch.tensor(loss_list).reshape(x.shape[0],x.shape[0]))
# plt.imshow(loss_list,extent=[0.2,2.0,0.2,2])
# plt.colorbar()
# plt.xlabel("tau0")
# plt.ylabel("tau1")
# plt.title("log(Loss) wtr to delay values")
# plt.pause(2)
# plt.close()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x, y, loss_list, lw=0.1, cmap='coolwarm', edgecolor='k')
# fig.colorbar(surf)
# ax.contour(x, y, loss_list, zdir='z', cmap='coolwarm')
# plt.tight_layout()
# plt.pause(2)
# plt.close()

# learnable_delays =  torch.abs(torch.randn((len(list_delays),)))
# model = SimpleNDDE2(dim=1, list_delays=learnable_delays)
# model = model.to(device)
# lossfunc = nn.MSELoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
# losses = []
# lens = []

# max_epoch = 10000
# for i in range(max_epoch):
#     # print(model.linear.weight)
#     model.linear.weight.requires_grad = False
#     opt.zero_grad()
#     ret = ddesolve_adjoint(history_function, model, ts)
#     loss = lossfunc(ret, ys)
#     loss.backward()
#     opt.step()
#     if i % 15 == 0:
#         for i in range(ys.shape[0]):
#             plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
#             plt.plot(ret[i].cpu().detach().numpy(), "--")
#         plt.legend()
#         plt.savefig("last_res.png", bbox_inches="tight", dpi=100)
#         plt.close()
#     print(
#         "Epoch : {:4d}, Loss : {:.3e}, tau : {}".format(
#             i, loss.item(), [p.item() for p in model.delays]
#         )
#     )

#     losses.append(loss.item())
#     idx = np.random.randint(0, ys.shape[0])
#     if losses[-1] < 1e-5 or i == max_epoch - 1:
#         plt.plot(ys[idx].cpu().detach().numpy())
#         plt.plot(ret[idx].cpu().detach().numpy(), "--")
#         plt.pause(2)
#         plt.close()
#         break


""" 
The second experiment is also with one delay but we have a history function that is not constant so x'(t) != 0 for t < 0. This helps us see
that the adjoint method is correct for the learning of delays.

We note the user that the adjoint method is needs to have the same solver for the forward and backward pass in order to have a good optimization.
Else there might be some discrepencies between the optimal delay and the one learned by the adjoint method.
"""

ys_true = ys.clone()
idx = (ts > 2.1).nonzero().flatten()[0] 
ts_history_train, ts_train = ts[:idx+1], ts[idx:]
ys_history, ys = ys_true[:, :idx+1], ys_true[:, idx:]   
history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
history_function = lambda t: history_interpolator(t)

x, y = np.meshgrid(np.linspace(0.2, 2.0, 50),np.linspace(0.2, 2.0, 50))
possible_delays = torch.from_numpy(np.concatenate([x.reshape(1,-1),y.reshape(1,-1)],axis=0).T)
loss_list = []
for delay in possible_delays:
    dde_solver = DDESolver(Euler(), delay)
    ys_other, _ = dde_solver.integrate(simple_dde2, ts_train, history_function)
    loss = torch.mean((ys -ys_other)**2)
    loss_list.append(loss.item())

loss_list = torch.log(torch.tensor(loss_list).reshape(x.shape[0],x.shape[0]))
plt.imshow(loss_list,extent=[0.2,2.0,0.2,2])
plt.colorbar()
plt.xlabel("tau0")
plt.ylabel("tau1")
plt.title("log(Loss) wtr to delay values")
plt.pause(2)
plt.close()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, loss_list, lw=0.1, cmap='coolwarm', edgecolor='k')
fig.colorbar(surf)
ax.contour(x, y, loss_list, zdir='z', cmap='coolwarm')
plt.tight_layout()
plt.pause(2)
plt.close()


learnable_delays = torch.abs(torch.randn((len(list_delays),)))
model = SimpleNDDE2(dim=1, list_delays=learnable_delays)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
losses = []
lens = []

max_epoch = 10000
for i in range(max_epoch):
    # print(model.linear.weight)
    model.linear.weight.requires_grad = False
    opt.zero_grad()
    ret = ddesolve_adjoint(history_function, model, ts_train)
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
        "Epoch : {:4d}, Loss : {:.3e}, tau : {}".format(
            i, loss.item(), [p.item() for p in model.delays]
        )
    )

    losses.append(loss.item())
    idx = np.random.randint(0, ys.shape[0])
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[idx].cpu().detach().numpy())
        plt.plot(ret[idx].cpu().detach().numpy(), "--")
        plt.pause(2)
        plt.close()
        break
