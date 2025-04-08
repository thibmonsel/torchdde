# Neural DDE

!!! warning

    This library only supports constant lag DDEs. Therefore we are unable to model time and state dependent DDEs.

This examples trains a Neural DDE with learnable delays to reproduce a simple dataset of a delay logistic equation. In this example, "discretize then optimize" is used to train the model. 

```python
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchdde import integrate, AdaptiveStepSizeController, Dopri5
from torchvision.ops import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Recalling that a Neural DDE is defined as

$$\frac{dy}{dt} = f_{\theta}(t, y(t), y(t-\tau_1), \dots, y(t-\tau_{n})), \quad y(t<0) = \psi(t)$$

then here we're now about to define $f_{\theta}$ that appears on that right hand side on the equation above

```python
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
        self.delays = delays
        self.mlp = MLP(
            self.in_dim,
            hidden_channels=depth * [width_size] + [out_size],
        )

    def forward(self, t, z, func_args, *, history):
        # `history` corresponds to the list of
        # delayed states defined in your DDE
        # i.e. here history=[y(t-tau1), ..., y(t-taun)]
        return self.mlp(torch.cat([z, *history], dim=-1))
```

We generate the toy dataset of the [delayed logistic equation](https://www.math.miami.edu/~ruan/MyPapers/Ruan-nato.pdf) (Equation 2.1).

```python
def get_data(y0, ts, tau=torch.tensor([1.0])):
    def f(t, y, func_args, history):
        return y * (1 - history[0])

    history_function = lambda t: y0
    ys = integrate(
        f,
        Dopri5(),
        ts[0],
        ts[-1],
        ts,
        history_function,
        func_args=None,
        stepsize_controller=AdaptiveStepSizeController(1e-6, 1e-9),
        dt0=ts[1] - ts[0],
        delays=tau,
    )
    return ys


class MyDataset(Dataset):
    def __init__(self, ys):
        self.ys = ys

    def __getitem__(self, index):
        return self.ys[index]

    def __len__(self):
        return self.ys.shape[0]
```

Main entry point. Try running `main()`.

```python

def main(
    dataset_size=128,
    batch_size=128,
    lr=0.01,
    max_epoch=100,
    width_size=32,
    depth=2,
    seed=5678,
    plot=True,
    print_every=5,
    device=device,
):
    torch.manual_seed(seed)
    ts = torch.linspace(0, 10, 101)
    y0_min, y0_max = 2.0, 3.0
    y0 = (y0_min - y0_max) * torch.rand((dataset_size,1)) + y0_max
    ys = get_data(y0, ts)
    ts, ys = ts.to(device), ys.to(device)
    delay_min, delay_max = 0.7, 1.3
    value = (delay_max - delay_min) * torch.rand((1,)) + delay_min
    # This allows for tau to be learnable and train along with the model.
    tau = torch.nn.Parameter(value)

    state_dim = ys.shape[-1]
    model = NDDE(tau, state_dim, state_dim, width_size, depth)
    # uncomment this line to make tau not learnable i.e. fixed
    # model.delays.requires_grad = False
    model = model.to(device)

    dataset = MyDataset(ys)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop like normal.
    losses, delays_evol = [], []
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max_epoch):
        model.train()
        for step, data in enumerate(train_loader):
            t = time.time()
            optimizer.zero_grad()
            data = data.to(device)
            history_fn = lambda t: data[:, 0]
            ys_pred = integrate(
                model,
                Dopri5(),
                ts[0],
                ts[-1],
                ts,
                history_fn,
                func_args=None,
                dt0=ts[1] - ts[0],
                stepsize_controller=AdaptiveStepSizeController(1e-6, 1e-9),
                discretize_then_optimize=True,
                delays=tau,
            )
            loss = loss_fn(ys_pred, data)
            losses.append(loss.item())
            delays_evol.append(model.delays.clone())
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
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(
            ts.cpu(),
            ys_pred[0].cpu().detach(),
            "--",
            c="crimson",
            label="Model",
        )
        plt.plot(
            ts.cpu(),
            data[0].cpu().detach(),
            label="Ground Truth",
        )
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(losses)
        plt.yscale("log")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.subplot(3, 1, 3)
        plt.plot(torch.stack(delays_evol).cpu().detach().numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Tau")
        plt.tight_layout()
        plt.savefig("neural_dde.png")
        plt.close()

    return ts, ys, model

ts, ys, model = main()
```
