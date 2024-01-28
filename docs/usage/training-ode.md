# Training a ODE

First, there are a lot of available package to use to train Neural ODEs, [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (not maintained anymore) in Pytorch and [Diffrax](https://github.com/patrick-kidger/diffrax). This means that this library doesn't have any many features since it focuses more on DDEs.

Two following ways are possible to train Neural ODE :

- optimize-then-discretize
- discretize-then-optimize

Please see the doctorial thesis [On Neural Differential Equations](https://arxiv.org/pdf/2202.02435.pdf) for more information on both procedures.

## optimize-then-discretize

If you choose to train with the adjoint method then you only to use `odesolve_adjoint` :

```python
import torch
from torchdde import odesolve_adjoint

history_function = torch.Tensor([...])
ts = torch.linspace(...)
pred = odesolve_adjoint(y0, model, ts, args, solver)
```

::: torchdde.odesolve_adjoint

## discretize-then-optimize

If you choose to train with the inherent auto differentiation capabilities of Pytorch then you need to use `AbstractOdeSolver` with specified : initial condition `y0`, pytorch model `model` and integration span `ts`.

```python
import torch
from torchdde import odesolve_adjoint

ode_solver = ...
y0 = ...
ts = torch.linspace(...)
pred = ode_solver.integrate(model, ts, y0, args)
```
