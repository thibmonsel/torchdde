import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, ddesolve_adjoint)


class SimpleNDDE(nn.Module):
    def __init__(self, dim, list_delays):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays =  nn.Parameter(list_delays)
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)
        self.init_weight()
        
    def init_weight(self):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[1.0, -1.0]]))

    def forward(self, t, z, *, history):
        z__history = z * history[0]
        inp = torch.cat([z, z__history], dim=-1)
        return self.linear(inp)



def simple_dde(t, y, *, history):
    return y * (1 - history[0])


def simple_dde2(t, y, *, history):
    return 1/2 * history[0] - history[1]


device = "cpu"
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_function = lambda t: history_values 
print("history_values", history_values.shape)

ts = torch.linspace(0, 20, 201)
list_delays = [1.0]
# list_delays = [1.0, 2.0]
solver = RK4()
dde_solver = DDESolver(solver, list_delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
print(ys.shape)

for i in range(ys.shape[0]):
    plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
plt.pause(2)
plt.close()

# 2 delays for brusselator looks like a good choice
learnable_delays =  torch.abs(torch.randn((len(list_delays),)))
model = SimpleNDDE(dim=1, list_delays=learnable_delays)
model = model.to(device)
lossfunc = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
losses = []
lens = []

max_epoch = 10000
for i in range(max_epoch):
    model.linear.weight.requires_grad = False
    opt.zero_grad()
    ret = ddesolve_adjoint(history_function, model, ts)
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
        plt.show()
        break
