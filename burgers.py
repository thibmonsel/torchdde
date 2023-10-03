
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from dataset import burgers
from model import NDDE, ConvNDDE
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, nddesolve_adjoint)

dataset_size = 32
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

max_delay = torch.tensor([2.0])
list_delays = torch.abs(torch.rand((1,)))
list_delays = torch.min(list_delays, max_delay.item() * torch.ones_like(list_delays))
model = ConvNDDE(1, list_delays)

model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
losses = []

# computing history function 
idx = (ts >= max_delay).nonzero().flatten()[0]
ts_history, ts = ts[:idx+1], ts[idx:]
ys_history, ys = ys[:, :idx+1], ys[:, idx:]   

ys_history_true, ys_true = ys_history.clone(), ys.clone()
ys_history_freq = torch.fft.rfft(ys_history, axis=1)
ys_history_freq[:, 4:, :] = 0
ys_history = torch.fft.irfft(ys_history_freq, axis=1, n=ts_history.shape[0])

ys_freq = torch.fft.rfft(ys, axis=1)
ys_freq[:, 4:, :] = 0
ys = torch.fft.irfft(ys_freq, axis=1, n=ts.shape[0])

plt.subplot(1, 2, 1)
plt.imshow(ys_history_true[j].cpu().detach().numpy(), label="Truth")
plt.subplot(1, 2, 2)
plt.imshow(ys_history[j].cpu().detach().numpy(), label="Truth")
plt.colorbar()
plt.pause(2)
plt.close() 


plt.subplot(1, 2, 1)
plt.imshow(ys_true[j].cpu().detach().numpy(), label="Truth")
plt.subplot(1, 2, 2)
plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
plt.colorbar()
plt.pause(2)
plt.close() 

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
    
    if losses[-1] < 1e-8 or i == max_epoch - 1:
        plt.subplot(1, 2, 1)
        plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
        plt.subplot(1, 2, 2)
        plt.imshow(ret[j].cpu().detach().numpy())
        plt.savefig('last_res.png',bbox_inches='tight',dpi=100)
        plt.close()
