from typing import Any, Callable, Union

import torch
from jaxtyping import Float
from torchdde.solver.base import AbstractOdeSolver


class Ralston(AbstractOdeSolver):
    """Ralston's method (2nd order)"""

    def __init__(self):
        super().__init__()

    def init(self):
        pass

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        if has_aux:
            k1, aux = func(t, y, args)
            k2, _ = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), None
