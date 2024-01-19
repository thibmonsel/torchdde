import torch
from torchdde import DDESolver, RK2


def f(t, y, *, history):
    return y * (1 - history[0])


delays = torch.tensor([1.0])
solver = DDESolver(RK2(), delays)
history_values = torch.arange(1, 5).reshape(-1, 1)
history_function = lambda t: history_values
solution, _ = solver.integrate(f, torch.linspace(0, 20, 201), history_function)
