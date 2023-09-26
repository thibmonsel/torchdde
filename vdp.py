
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from dataset import brusellator, stiff_vdp
from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, nddesolve_adjoint)

dataset_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ts = torch.linspace(0, 30.0, 301)
y0 = np.random.uniform(0.1, 2.0, (dataset_size, 2))
y0[:, 0] = 0.0

ys = stiff_vdp(y0, ts)
ys = ys.to(torch.float32)
ys, ts = ys.to(device), ts.to(device)

for i in range(ys.shape[0]):
    plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close() 

# workable list of delays : [1.4, 2.8], [1.0, 2.0], [3.0]
list_delays = [2.5]
model = NDDE(ys.shape[-1], list_delays, width=128)

model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=10e-3, weight_decay=0)
losses = []

# computing history function 
max_delay = max(list_delays)
idx = (ts == max_delay).nonzero().flatten()
ts_history, ts = ts[:idx+1], ts[idx:]
ys_history, ys = ys[:, :idx+1], ys[:, idx:]
history_interpolator = TorchLinearInterpolator(ts_history, ys_history)
history_function = lambda t: history_interpolator(t)

max_epoch = 5000
for i in range(max_epoch):
    opt.zero_grad()
    t = time.time()
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 15 == 0:
        for i in range(ys.shape[0]):
            plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
            plt.plot(ret[i].cpu().detach().numpy(), "--")
        plt.savefig('last_res.png',bbox_inches='tight',dpi=100)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}, Time : {:.2e}".format(i, loss.item(), time.time() - t))

    losses.append(loss.item())
    
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break

