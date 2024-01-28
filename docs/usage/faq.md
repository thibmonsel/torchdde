# FAQ

## How would I define a DDE with several delays ?

You just have to specify a delay tensor size that corresponds to the number of delays you desire.

```python
solver = ....
delays = torch.tensor([1.0, 2.0])
history_function = lambda t : ...
ts = ...

def simple_dde(t, y, args, *, history):
    # this correspond to y'(t) = -y(t-1) - y(t-2)
    return - history[0] - history[1]

dde_solver = DDESolver(solver, delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
```

## How about if I want a neural network to have also several delays ?

Well if its the same forward pass in the [Neural DDE](./neural-dde.md), then nothing needs to be changed ! The term `torch.cat([z, *history], dim=-1)` unpacks all the delayed terms.
