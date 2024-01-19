# Neural DDE

!!! warning

    This library only supports constant lag DDEs. Therefore we are unable to model state dependent delays.

This examples trains a Neural DDE to reproduce a simple dataset of a delay logistic equation. The backward pass is compute with the adjoint method i.e `ddesolve_adjoint`.

```python
import torch
from torchvision.ops import MLP
from torchdde import DDESolver, Euler

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Recalling that a neural DDE is defined as

$$ \frac{dy}{dt} = f_{\theta}(t, y(t), y(t-\tau_1), \dots, y(t-\tau_{n})), \quad
\\ x(t<0) = \psi(t)$$

then here we're now about to define $f_{\theta}$ that appears on that right hand side

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
        self.delays = torch.nn.Parameter(delays)
        self.mlp = MLP(
            self.in_dim,
            hidden_channels=depth * [width_size] + [out_size],
        )

    def forward(self, t, z, *, history):
        return self.mlp(torch.cat([z, *history], dim=-1))
```

We generate the toy dataset of the [delayed logistic equation](https://www.math.miami.edu/~ruan/MyPapers/Ruan-nato.pdf) (Equation 2.1).

```python
def get_data(y0, ts, tau=torch.tensor([1.0])):
    def f(t, y, history):
        return y * (1 - history[0])

    solver = DDESolver(Euler(), tau)
    ys, _ = solver.integrate(f, ts, lambda t: torch.unsqueeze(y0, dim=1))
    return ts, ys

dataset_size = 256
state_dim = 1
ts = torch.linspace(0, 10, 101)
r1, r2 = 2.0, 3.0
y0 = (r1 - r2) * torch.rand((dataset_size, state_dim)) + r2
ys = get_data(y0, ts)

value = torch.abs(torch.rand((1,)))
list_delays = torch.tensor([value])
list_delays = list_delays.to(device)

model = NDDE(list_delays, state_dim, state_dim, 32, 2)
model = model.to(device)

dataset = MyDataset(ys)
train_len = int(len(dataset) * 0.7)
train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
```

Main entry point. Try running `main()`.

```python
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
```
