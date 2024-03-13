from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver


torch.set_default_dtype(torch.float64)


class Dopri5(AbstractOdeSolver):
    """5th order order explicit Runge-Kutta method"""

    a_lower = (
        (
            torch.tensor([1 / 5], dtype=torch.float64),
            torch.tensor([3 / 40, 9 / 40], dtype=torch.float64),
            torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64),
            torch.tensor(
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
                dtype=torch.float64,
            ),
            torch.tensor(
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
                dtype=torch.float64,
            ),
            torch.tensor(
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
                dtype=torch.float64,
            ),
        ),
    )
    b_sol = (
        torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            dtype=torch.float64,
        ),
    )
    b_error = (
        torch.tensor(
            [
                35 / 384 - 1951 / 21600,
                0,
                500 / 1113 - 22642 / 50085,
                125 / 192 - 451 / 720,
                -2187 / 6784 - -12231 / 42400,
                11 / 84 - 649 / 6300,
                -1.0 / 60.0,
            ],
            dtype=torch.float64,
        ),
    )
    c = (torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=torch.float64),)

    def __init__(self):
        super().__init__()

    def init(self):
        pass

    def order(self):
        return 5

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
        Float[torch.Tensor, " batch"],
        dict[str, Float[torch.Tensor, "batch order"]],
        Union[Float[torch.Tensor, " batch"], Any],
    ]:
        if has_aux:
            k = []
            k1, aux = dt * func(t, y, args)
            k.append(k1)
            for ci, ai in zip(self.c[0], self.a_lower[0]):
                ki = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            y_error = y + dt * torch.einsum(
                "k, kbf -> bf", self.b_error[0], torch.stack(k)
            )
            dense_info = dict(y0=y, y1=y1, k=torch.concat(k, dim=-1))
            return y1, y_error, dense_info, aux
        else:
            k = []
            k.append(dt * func(t, y, args))
            for ci, ai in zip(self.c[0], self.a_lower[0]):
                ki = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            z1 = y + dt * torch.einsum("k, kbf -> bf", self.b_error[0], torch.stack(k))
            y_error = torch.abs(z1 - y1)
            dense_info = dict(y0=y, y1=y1, k=torch.concat(k[1:-1], dim=-1))
            return y1, y_error, dense_info, None
