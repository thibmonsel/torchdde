
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from dataset import ks
from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, nddesolve_adjoint)

dataset_size = 32
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
ts = torch.linspace(0, 30, 301)

ys = ks(dataset_size, ts)
ys = ys.to(torch.float32)
ys, ts = ys.to(device), ts.to(device)
print(ys.shape)

j = np.random.randint(0, dataset_size)
plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close() 

max_delay = torch.tensor([5.0])
list_delays = torch.abs(torch.rand((1,)))
list_delays = torch.min(list_delays, max_delay.item() * torch.ones_like(list_delays))
model = NDDE(ys.shape[-1], list_delays, width=2*258)

model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
losses = []

# computing history function 
idx = (ts >= max_delay).nonzero().flatten()[0]
ts_history, ts = ts[:idx+1], ts[idx:]
ys_history, ys = ys[:, :idx+1], ys[:, idx:]   
history_interpolator = TorchLinearInterpolator(ts_history, ys_history)
history_function = lambda t: history_interpolator(t)


max_epoch = 10000
for i in range(max_epoch):
    opt.zero_grad()
    t = time.time()
    ret = nddesolve_adjoint(history_function, model, ts)
    loss = lossfunc(ret, ys)
    loss.backward()
    opt.step()
    if i % 15 == 0:
        plt.subplot(1, 2, 1)
        plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(ret[j].cpu().detach().numpy())
        plt.colorbar()
        plt.savefig('last_res.png',bbox_inches='tight',dpi=100)
        plt.close()
    print("Epoch : {:4d}, Loss : {:.3e}, tau : {}".format(i, loss.item(), [d.item() for d in model.delays]))

    losses.append(loss.item())
    
    if losses[-1] < 1e-5 or i == max_epoch - 1:
        plt.plot(ys[0].cpu().detach().numpy())
        plt.plot(ret[0].cpu().detach().numpy(), "--")
        plt.show()
        break

