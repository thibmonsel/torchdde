from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import ThirdOrderPolynomialInterpolation


class Ralston(AbstractOdeSolver):
    """2nd order Ralston's method"""

    interpolation_cls = ThirdOrderPolynomialInterpolation

    def __init__(self):
        super().__init__()

    def init(self):
        pass

    def order(self):
        return 2

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
            k2, _ = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            y1 = y + dt * (1 / 4 * k1 + 3 / 4 * k2)
            return y1, None, dict(y0=y, y1=y1, k=torch.stack([k1, k2])), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            y1 = y + dt * (1 / 4 * k1 + 3 / 4 * k2)
            return y1, None, dict(y0=y, y1=y1, k=torch.stack([k1, k2])), None

    def build_interpolation(
        self, t0, t1, dense_info
    ) -> ThirdOrderPolynomialInterpolation:
        return self.interpolation_cls(t0, t1, dense_info)
