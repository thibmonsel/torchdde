# Getting Started

## Integrating ...
### ... DDEs

We provide an illustrative example which solves the following DDE :

$\frac{dy}{dt}= -y(t-2), \quad \psi(t<0) = 2$ over the interval $[0, 5]$.

```python
import matplotlib.pyplot as plt
from torchdde import RK4
import torch

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
solution = torchdde.integrate(simple_dde, solver, ts[0], ts[-1], ts, history_function, None, dt0=ts[1]-ts[0], delays=delays)
```

- The numerical solver used is `RK4` is the 4th Runge Kutta method.
- The solution is saved at each time stamp in `ts`.
- The initial step size is equal to `ts[1]-ts[0]`.

### ... ODEs

We provide an illustrative example which solves the following ODE :

$\frac{dy}{dt}= -y(t)^{2}, \quad y(0) = 2$ over the interval $[0, 5]$.

```python
import matplotlib.pyplot as plt
from torchdde import RK4
import torch

def simple_ode(t, y, args):
    return -y**2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the time span of the ODE
ts = torch.linspace(0, 5, 51)
ts = ts.to(device)

# Initial condition 
y0 = torch.tensor([3.0])
y0 = y0.reshape(-1, 1)
y0 = y0.to(device)

# Solve the ODE by using the RK4 method
solver = RK4()
solution = torchdde.integrate(simple_ode, solver, ts[0], ts[-1], ts, y0, None, dt0=ts[1]-ts[0], delays=None)
```

- The numerical solver used is `RK4` is the 4th Runge Kutta method.
- The solution is saved at each time stamp in `ts`.
- The initial step size is equal to `ts[1]-ts[0]`.
