from typing import Any, Callable, Union

import torch
from jaxtyping import Float
from torchdde.solver.base import AbstractOdeSolver


class Euler(AbstractOdeSolver):
    """Euler's method"""

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
            return y + dt * k1, aux
        else:
            return y + dt * func(t, y, args), None
