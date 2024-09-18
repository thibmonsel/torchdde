# torchdde

`torchdde` is a library that provides numerical solvers in [Pytorch](https://github.com/pytorch/pytorch) for Delay Differential Equations (DDEs) with constant delays.

## Installation

```bash
pip install git@github.com:thibmonsel/torchdde.git
```

or locally

```bash
git clone https://github.com/thibmonsel/torchdde.git
pip install torchdde/
```

!!! warning

    This a brand new library, please reach out for feedback, issues !


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
