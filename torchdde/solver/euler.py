from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import FirstOrderPolynomialInterpolation


class Euler(AbstractOdeSolver):
    """Euler's method"""

    interpolation_cls = FirstOrderPolynomialInterpolation

    def __init__(self):
        super().__init__()

    def init(self):
        pass

    def order(self):
        return 1

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Float[torch.Tensor, ""],
        args: Any,
        has_aux=False,
    ) -> tuple[
        Float[torch.Tensor, "batch ..."],
        Any,
        dict[str, Float[torch.Tensor, "batch order"]],
        Union[Float[torch.Tensor, " batch"], Any],
    ]:
        if has_aux:
            k1, aux = func(t, y, args)
            y1 = y + dt * k1
            return y1, None, dict(y0=y, y1=y1), aux
        else:
            y1 = y + dt * func(t, y, args)
            return y1, None, dict(y0=y, y1=y1), None

    def build_interpolation(self, t0, t1, dense_info):
        return self.interpolation_cls(t0, t1, dense_info)
