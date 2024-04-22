from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import FirstOrderPolynomialInterpolation


class Heun(AbstractOdeSolver):
    """2th order order explicit Runge-Kutta method Euler-Heun"""

    # add another dim to a_lower to make einsum work on ki
    a_lower = (torch.tensor([[1.0]]),)
    b_sol = (torch.tensor([0.5, 0.5]),)
    b_error = (torch.tensor([0.5, -0.5]),)
    c = (torch.tensor([1.0]),)

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
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[
        Float[torch.Tensor, "batch ..."],
        Float[torch.Tensor, "batch ..."],
        dict[str, Float[torch.Tensor, "..."]],
        Union[Float[torch.Tensor, "batch ..."], Any],
    ]:
        if has_aux:
            k = []
            k1, aux = func(t, y, args)
            k.append(k1)
            for ci, ai in zip(self.c[0], self.a_lower[0]):
                ki, _ = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            y_error = torch.einsum("k, kbf -> bf", self.b_error[0], dt * torch.stack(k))
            dense_info = dict(y0=y, y1=y1, k=torch.stack(k))
            return y1, y_error, dense_info, aux
        else:
            k = []
            k.append(func(t, y, args))
            for ci, ai in zip(self.c[0], self.a_lower[0]):
                ki = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            y_error = torch.einsum(
                "k, kbf -> bf", self.b_error[0], dt * torch.stack(k)
            ).abs()
            dense_info = dict(y0=y, y1=y1, k=torch.stack(k))
            return y1, y_error, dense_info, None

    def build_interpolation(self, t0, t1, dense_info):
        return FirstOrderPolynomialInterpolation(t0, t1, dense_info)
