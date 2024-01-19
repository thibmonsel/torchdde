# Neural DDE
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchdde import ddesolve_adjoint, DDESolver, Euler
from torchvision.ops import MLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NDDE(nn.Module):
    def __init__(
        self,
        delays,
        in_size,
        out_size,
        width_size,
        depth,
    ):
        super().__init__()
        self.in_dim = in_size * (1 + len(delays))
        self.delays = torch.nn.Parameter(delays)
        self.mlp = MLP(
            self.in_dim,
            hidden_channels=depth * [width_size] + [out_size],
        )

    def forward(self, t, z, *, history):
        return self.mlp(torch.cat([z, *history], dim=-1))


def get_data(y0, ts, tau=torch.tensor([1.0])):
    def f(t, y, history):
        return y * (1 - history[0])

    solver = DDESolver(Euler(), tau)
    ys, _ = solver.integrate(f, ts, lambda t: torch.unsqueeze(y0, dim=1))
    return ys


class MyDataset(Dataset):
    def __init__(self, ys):
        self.ys = ys

    def __getitem__(self, index):
        return self.ys[index]

    def __len__(self):
        return self.ys.shape[0]


def main(
    dataset_size=128,
    batch_size=128,
    lr=0.0005,
    max_epoch=1000,
    width_size=32,
    depth=2,
    seed=5678,
    plot=True,
    print_every=5,
):
    torch.manual_seed(seed)
    ts = torch.linspace(0, 10, 101)
    y0_min, y0_max = 2.0, 3.0
    y0 = (y0_min - y0_max) * torch.rand((dataset_size,)) + y0_max
    ys = get_data(y0, ts)
    ts, ys = ts.to(device), ys.to(device)

    delay_min, delay_max = 0.7, 1.3
    value = (delay_max - delay_min) * torch.rand((1,)) + delay_min
    list_delays = torch.tensor([value])
    list_delays = list_delays.to(device)

    state_dim = ys.shape[-1]
    model = NDDE(list_delays, state_dim, state_dim, width_size, depth)
    model = model.to(device)

    dataset = MyDataset(ys)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop like normal.

    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max_epoch):
        for step, data in enumerate(train_loader):
            t = time.time()
            optimizer.zero_grad()
            data = data.to(device)
            history_fn = lambda t: data[:, 0]
            ys_pred = ddesolve_adjoint(history_fn, model, ts, Euler())
            loss = loss_fn(ys_pred, data)
            loss.backward()
            optimizer.step()
            if (epoch % print_every) == 0 or epoch == max_epoch - 1:
                print(
                    "Epoch : {}, Step {}/{}, Loss : {:.3e}, Tau {}, Time {}".format(
                        epoch,
                        step + 1,
                        len(train_loader),
                        loss.item(),
                        [d.item() for d in model.delays],
                        time.time() - t,
                    )
                )
    if plot:
        plt.plot(ts.cpu(), data[0].cpu(), c="dodgerblue", label="Real")
        history_values = data[0, 0][..., None]
        history_fn = lambda t: history_values
        ys_pred = ddesolve_adjoint(history_fn, model, ts, Euler())
        plt.plot(
            ts.cpu(),
            ys_pred[0].cpu().detach(),
            "--",
            c="crimson",
            label="Model",
        )
        plt.legend()
        plt.savefig("neural_dde.png")
        plt.show()
        plt.close()

    return ts, ys, model


ts, ys, model = main()