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

To generate the documentation, please look at `CONTRIBUTING.MD`.

## Quick example

```python
import torch
from torchdde import DDESolver, RK2

def f(t, y, history):
    return y * (1 - history[0])

delays = torch.tensor([1.0])
solver = DDESolver(RK2(), delays)
history_values = torch.arange(1, 5).reshape(-1, 1)
history_function = lambda t: history_values
solution, _ = solver.integrate(f, torch.linspace(0, 20, 201), history_function,None)

```
