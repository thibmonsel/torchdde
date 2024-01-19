# Getting Started

An illustrative example which solves the following DDE

$\frac{dy}{dt}= -y(t-2), \quad \psi(t<0) = 2$ over the interval $[0, 5]$.

```python
import torch
import matplotlib.pyplot as plt
from torchdde import DDESolver, RK4

def simple_dde(t, y, args, *, history):
    return - history[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the time span of the DDE
ts = torch.linspace(0, 5, 51)
ts = ts.to(device)

# Define the delays, here there is only one tau=2.0
list_delays = torch.tensor([2.0])
list_delays = list_delays.to(device)

# Defining a constant history function for the DDE
history_values = torch.tensor([3.0])
history_values = history_values.reshape(-1, 1)
history_values = history_values.to(device)
history_interpolator = lambda t : history_values

# Solve the DDE by using the RK4 method
dde_solver = DDESolver(RK4(), list_delays)
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)
```

- The numerical solver used is `RK4` is the 4th Runge Kutta method.
- The solution is saved at each time stamp in `ts`.
- The initial step size is equal to `ts[1]-ts[0]`.
