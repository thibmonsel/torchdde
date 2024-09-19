<h1 align='center'>torchdde</h1>
<!-- <h2 align='center'> Constant lag delay differential equations solver</h2> -->

torchdde is a library that provides numerical solvers in Pytorch for Delay Differential Equations (DDEs) with constant delays.

## Installation

```bash
pip install git@github.com:thibmonsel/torchdde.git
```

or locally

```bash
git clone https://github.com/thibmonsel/torchdde.git
pip install torchdde/
```

## Documentation

Github pages hosts the documentation at : [https://thibmonsel.github.io/torchdde/](https://thibmonsel.github.io/torchdde/)

To generate the documentation locally, please look at `CONTRIBUTING.MD`.

## Quick example

```python
import torch
from torchdde import integrate, RK2

def f(t, y, args, history):
    return y * (1 - history[0])

solver = RK2()
delays = torch.tensor([1.0])
history_values = torch.arange(1, 5).reshape(-1, 1)
history_function = lambda t: history_values
solution = integrate(f, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1]-ts[0], delays=delays)

```
