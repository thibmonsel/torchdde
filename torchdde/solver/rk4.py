from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import ThirdOrderPolynomialInterpolation


class RK4(AbstractOdeSolver):
    """4th order explicit Runge-Kutta method"""

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
            k2, _ = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3, _ = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4, _ = func(t + dt, y + dt * k3, args)
            y1 = y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
            return y1, None, dict(y0=y, k=torch.stack([k1, k2, k3, k4]), y1=y1), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4 = func(t + dt, y + dt * k3, args)
            y1 = y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
            return y1, None, dict(y0=y, k=torch.stack([k1, k2, k3, k4]), y1=y1), None

    def build_interpolation(
        self, t0, t1, dense_info
    ) -> ThirdOrderPolynomialInterpolation:
        return self.interpolation_cls(t0, t1, dense_info)
