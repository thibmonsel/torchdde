# Training a DDE

Two following ways are possible to train Neural DDE :

- optimize-then-discretize
- discretize-then-optimize

Please see the doctorial thesis [On Neural Differential Equations](https://arxiv.org/pdf/2202.02435.pdf) for more information on both procedures.

## optimize-then-discretize

If you choose to train with the adjoint method then you only to use `ddesolve_adjoint` :

```python
import torch
from torchdde import ddesolve_adjoint

history_function = lambda t : ...
ts = torch.linspace(...)
pred = ddesolve_adjoint(history_function, model, ts, solver)
```

::: torchdde.ddesolve_adjoint

## discretize-then-optimize

!!! warning

    You are unable to learn the DDE's delays if using the discretize-then-optimize approach.

If you choose to train with the inherent auto differentiation capabilities of Pytorch then you need to use `DDESolver` with specified : history function `history_function`, pytorch model `model`, integration span `ts`, ode solver used `solver`.

```python
import torch
from torchdde import ddesolve_adjoint

ode_solver = ...
tensor_delays = ...
dde_solver = DDEsolver(ode_solver, tensor_delays)
history_function = lambda t : ...
ts = torch.linspace(...)
pred, _ = dde_solver.integrate(model, ts, history_function)
```
