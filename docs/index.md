# torchdde

`torchdde` is a library providing Constant Lag Delay Differential Equations (DDEs) training neural networks via the adjoint method compatible with [Pytorch](https://github.com/pytorch/pytorch).

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

## Quick examples

`torchdde` can solve constant lag DDEs :
```python
import torchdde
import torch

def f(t, y, args, history):
    return y * (1 - history[0])

delays = torch.tensor([1.0])
history_values = torch.arange(1, 5).reshape(-1, 1)
history_function = lambda t: history_values
ts = torch.linspace(0, 20, 201)
solution = torchdde.integrate(
    f,
    torchdde.RK2(),
    ts[0],
    ts[-1],
    ts,
    history_function,
    None,
    dt0=ts[1] - ts[0],
    delays=delays,
)
```
