<h1 align='center'>torchdde</h1>
<!-- <h2 align='center'> Constant lag delay differential equations solver</h2> -->

`torchdde` is a [Pytorch](https://github.com/pytorch/pytorch)-based library providing Constant Lag Delay Differential Equations (DDEs) training neural networks via the adjoint method.

## Installation

```bash
pip install git@github.com:usr/torchdde.git
```

or locally

```bash
git clone https://github.com/usr/torchdde.git
cd torchdde/
pip install .
```

## Documentation

To generate the documentation :

```bash
mkdocs build
mkdocs build 
mkdocs serve 
```

## Quick example

```python
import torch
from torchdde import DDESolver, RK2

def f(t, y, history):
    return y * (1 - history[0])

solver = RK2()
delays = torch.tensor([1.0])
history_values = torch.arange(1, 5).reshape(-1, 1)
history_function = lambda t: history_values
solution = torchdde.integrate(f, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1]-ts[0], delays=delays)

```
