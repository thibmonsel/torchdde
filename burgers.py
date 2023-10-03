
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from dataset import burgers
from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, nddesolve_adjoint)

dataset_size = 16
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
ts = torch.linspace(0, 10, 101)
xs = np.linspace(0, 1, 200)

ys = burgers(dataset_size, ts, xs)
ys = ys.to(torch.float32)
ys, ts = ys.to(device), ts.to(device)
print(ys.shape)

j = np.random.randint(0, dataset_size)
plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
plt.colorbar()
plt.pause(2)
plt.close() 

nb_delays = 2
max_delay = torch.tensor([5.0])
list_delays = torch.abs(torch.rand((nb_delays,)))
list_delays = torch.min(list_delays, max_delay.item() * torch.ones_like(list_delays))

# computing history function 
nb_features = 10
features_idx = np.random.randint(0, ys.shape[-1], size=(nb_features, ))
idx = (ts >= max_delay).nonzero().flatten()[0]
ts_history, ts = ts[:idx+1], ts[idx:]
ys_history, ys = ys[:, :idx+1, features_idx], ys[:, idx:, features_idx]
history_interpolator = TorchLinearInterpolator(ts_history, ys_history)
history_function = lambda t: history_interpolator(t)

model = NDDE(ys.shape[-1], list_delays, width=128)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0)
losses = []

max_epoch = 10000
# solver = Ralston()
# dde_solver = DDESolver(solver, list_delays)
for i in range(max_epoch):
    opt.zero_grad()
    t = time.time()
    # ret, _ = dde_solver.integrate(model, ts, history_function)
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 15 == 0:
        plt.plot(ys[j].cpu().detach().numpy(), label="Truth")
        plt.plot(ret[j].cpu().detach().numpy(), '--')
        plt.savefig('last_res.png',bbox_inches='tight',dpi=100)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}, tau : {}".format(i, loss.item(), [d.item() for d in model.delays]))

    losses.append(loss.item())
    
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break

