import matplotlib.pyplot as plt
import torch
from torchdde import DDESolver, RK2, TorchLinearInterpolator


def simple_dde(t, y, args, *, history):
    """
    Vector Field of DDE equation
    y'(t) = y(t) * (1 - y(t - tau))

    Args:
        t (torch.Tensor): time t
        y (torch.Tensor): DDE's state
        history (Callable): DDE's history function

    Returns:
        (torch.Tensor): DDE's rhs
    """
    return y * (1 - history[0])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the time span of the DDE
ts = torch.linspace(0, 20, 201)
ts = ts.to(device)

# Define the delays, here there is only one tau=1.0
list_delays = torch.tensor([1.0])
list_delays = list_delays.to(device)

# Defining a constant history function for the DDE
# Psi(-10.0 < t < 0) = history_values
# We use the torchdde's linear interpolator class `TorchLinearInterpolator`
history_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
history_values = history_values.view(history_values.shape[0], 1)
history_interpolator = TorchLinearInterpolator(
    torch.tensor(([-10.0, 0.0])),
    torch.concat((history_values, history_values), dim=1)[..., None],
)
history_interpolator.to(device)
history_function = lambda t: history_interpolator(t)

# Solve the DDE by using the RK2 method
dde_solver = DDESolver(RK2(), list_delays)
# We specify :
# - the DDE's vector field
# - integration time span `ts`
# - the history function `history_function`
ys, _ = dde_solver.integrate(simple_dde, ts, history_function)

for i in range(ys.shape[0]):
    plt.plot(ts.cpu().detach().numpy(), ys[i].cpu().detach().numpy())
plt.title(f"y'(t) = y(t) * (1 - y(t - {list_delays[0]})) ")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.show()
plt.close()
