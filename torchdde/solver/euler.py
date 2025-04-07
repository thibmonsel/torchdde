from typing import Any, Callable, Dict, Tuple, Type, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.first_order_interpolation import (
    AbstractLocalInterpolation,
    FirstOrderPolynomialInterpolation,
)
from torchdde.solver.base import AbstractOdeSolver


class Euler(AbstractOdeSolver):
    """Euler's method"""

    interpolation_cls: Type[
        AbstractLocalInterpolation
    ] = FirstOrderPolynomialInterpolation

    def __init__(self):
        super().__init__()

    def init(self, func, t0, y0, dt0, func_args, *args, **kwargs):
        del func, t0, y0, dt0, func_args, args, kwargs
        return None

    def order(self):
        return 1

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Float[torch.Tensor, ""],
        solver_state: Union[Tuple[Any, ...], None],
        func_args: Any,
    ) -> Tuple[
        Float[torch.Tensor, "batch ..."],
        None,
        Dict[str, Float[torch.Tensor, "batch ..."]],
        None,
        Any,
    ]:
        assert solver_state is None, "Euler solver should be stateless"

        y1 = y + dt * func(t, y, func_args)
        return y1, None, dict(y0=y, y1=y1), None, None

    def build_interpolation(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "batch ..."]],
    ) -> FirstOrderPolynomialInterpolation:
        return self.interpolation_cls(t0, t1, dense_info)  # type: ignore
