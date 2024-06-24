from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import FourthOrderPolynomialInterpolation


class _Dopri5Interpolation(FourthOrderPolynomialInterpolation):
    c_mid = torch.tensor(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ],
    )

    def __init__(self, t0, t1, dense_info):
        super().__init__(t0, t1, dense_info, self.c_mid)


class Dopri5(AbstractOdeSolver):
    """5th order order explicit Runge-Kutta method Dormand Prince"""

    interpolation_cls = _Dopri5Interpolation

    a_lower = (
        torch.tensor([1 / 5]),
        torch.tensor([3 / 40, 9 / 40]),
        torch.tensor([44 / 45, -56 / 15, 32 / 9]),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
    )
    b_sol = (
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]),
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
        ),
    )
    c = (torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]),)

    def __init__(self):
        super().__init__()

    def init(self):
        pass

    def order(self):
        return 5

    def to(self, device):
        self.a_lower = tuple([ai.to(device) for ai in self.a_lower])
        self.b_sol = tuple([bi.to(device) for bi in self.b_sol])
        self.b_error = tuple([be.to(device) for be in self.b_error])
        self.c = tuple([ci.to(device) for ci in self.c])
        self.interpolation_cls.c_mid = self.interpolation_cls.c_mid.to(device)

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
            for ci, ai in zip(self.c[0], self.a_lower):
                ki, _ = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            y_error = dt * torch.einsum("k, kbf -> bf", self.b_error[0], torch.stack(k))
            dense_info = dict(y0=y, y1=y1, k=torch.stack(k))
            return y1, y_error, dense_info, aux
        else:
            k = []
            k.append(func(t, y, args))
            for ci, ai in zip(self.c[0], self.a_lower):
                ki = func(
                    t + dt * ci,
                    y + dt * torch.einsum("k, kbf -> bf", ai, torch.stack(k)),
                    args,
                )
                k.append(ki)
            y1 = y + dt * torch.einsum("k, kbf -> bf", self.b_sol[0], torch.stack(k))
            y_error = dt * torch.einsum("k, kbf -> bf", self.b_error[0], torch.stack(k))
            dense_info = dict(y0=y, y1=y1, k=torch.stack(k))
            return y1, y_error, dense_info, None

    def build_interpolation(self, t0, t1, dense_info):
        return self.interpolation_cls(t0, t1, dense_info)
