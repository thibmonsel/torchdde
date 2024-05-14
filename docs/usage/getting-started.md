# Getting Started

An illustrative example which solves the following DDE :

$\frac{dy}{dt}= -y(t-2), \quad \psi(t<0) = 2$ over the interval $[0, 5]$.

```python
import matplotlib.pyplot as plt
import torch
from torchdde import DDESolver, RK4


def simple_dde(t, y, args, *, history):
    return -history[0]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the time span of the DDE
ts = torch.linspace(0, 5, 51)
ts = ts.to(device)

# Define the delays, here there is only one tau=2.0
delays = torch.tensor([2.0])
delays = delays.to(device)

# Defining a constant history function for the DDE
history_values = torch.tensor([3.0])
history_values = history_values.reshape(-1, 1)
history_values = history_values.to(device)
history_function = lambda t: history_values

# Solve the DDE by using the RK4 method
solver = RK4()
solution = torchdde.integrate(f, solver, ts[0], ts[-1], ts, history_function, None, dt0=ts[1]-ts[0], delays=delays)
```

- The numerical solver used is `RK4` is the 4th Runge Kutta method.
- The solution is saved at each time stamp in `ts`.
- The initial step size is equal to `ts[1]-ts[0]`.
