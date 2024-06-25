from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.solver.base import AbstractOdeSolver

from ..local_interpolation import FirstOrderPolynomialInterpolation


class ImplicitEuler(AbstractOdeSolver):
    """ImplicitEuler Euler's method"""

    # Credits to the TorchDyn team for the implementation
    # of the implicit Euler method, adapted from:
    # https://github.com/DiffEqML/torchdyn/blob/95cc74b0e35330b03d2cd4d875df362a93e1b5ea/torchdyn/numerics/solvers/ode.py#L181

    interpolation_cls = FirstOrderPolynomialInterpolation

    def __init__(self, max_iters=100):
        super().__init__()
        self.opt = torch.optim.LBFGS
        self.max_iters = max_iters

    def init(self):
        pass

    def order(self):
        return 1

    @staticmethod
    def _residual(
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Float[torch.Tensor, ""],
        y_sol: Float[torch.Tensor, "batch ..."],
        args: Any,
        has_aux=False,
    ) -> Float[torch.Tensor, ""]:
        if has_aux:
            f_sol, _ = func(t, y_sol, args)
        else:
            f_sol = func(t, y_sol, args)
        return torch.sum((y_sol - y - dt * f_sol) ** 2)

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
        y_sol = y.clone()
        y_sol = torch.nn.Parameter(data=y_sol)
        opt = self.opt(
            [y_sol],
            lr=1,
            max_iter=self.max_iters,
            max_eval=10 * self.max_iters,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Float[torch.Tensor, ""]:
            opt.zero_grad()
            residual = ImplicitEuler._residual(
                func, t, y, dt, y_sol, args, has_aux=has_aux
            )
            (y_sol.grad,) = torch.autograd.grad(
                residual, y_sol, only_inputs=True, allow_unused=False
            )
            return residual

        opt.step(closure)  # type: ignore
        if has_aux:
            _, aux = func(t, y, args)
            return y_sol, None, dict(y0=y, y1=y_sol), aux
        else:
            return y_sol, None, dict(y0=y, y1=y_sol), None

    def build_interpolation(
        self, t0, t1, dense_info
    ) -> FirstOrderPolynomialInterpolation:
        return self.interpolation_cls(t0, t1, dense_info)
