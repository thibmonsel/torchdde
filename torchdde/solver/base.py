from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.base import AbstractInterpolation


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the `init`, `step` and `order` method.
    """

    interpolation_cls: AbstractInterpolation

    @abstractmethod
    def init(self):
        """
        Initializes the solver. This method is called before the integration starts.
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """
        Returns the order of the solver.
        """
        pass

    @abstractmethod
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
        Float[torch.Tensor, "batch ..."],
        dict[str, Float[torch.Tensor, "batch order"]],
        Union[Float[torch.Tensor, " batch"], Any],
    ]:
        """ODE's solver stepping method

        **Arguments:**

        - `func`: Pytorch model or callable function, i.e vector field
        - `t`: Current time step `t`
        - `y`: Current state `y`
        - `dt`: Step size `dt`
        - `has_aux`: Whether the model/callable has an auxiliary output.

        ??? tip "has_aux ?"

            A function with an auxiliary output can look like
            ```python
            def f(t,y,args):
                return -y, ("Hello World",1)
            ```
            The `has_aux` `kwargs` argument is used to compute the adjoint method

        **Returns:**

        - The value of the solution at `t+dt`.
        - A local error estimate made during the step. (Used by
        [`torchdde.AdaptiveStepSizeController`][] controllers to change the step size.)
        It may be `None` for constant stepsize solvers for example.
        - Dictionary that holds all the information needed to properly
        build the interpolation between `t` and `t+dt`.
        - None if the model doesn't have an auxiliary output.
        """
        pass

    @abstractmethod
    def build_interpolation(self, t0, t1, dense_info) -> Any:
        """Interpolator building method based on the solver's order.

        **Arguments:**

        - `t0`: The start of the interval over which the interpolation is defined.
        - `t1`: The end of the interval over which the interpolation is defined.
        - `dense_info`: Dictionary that hold all the information needed to properly
        build the interpolation between `t` and `t+dt`.

        **Returns:**

        A `Callable` that can be used to interpolate the solution between `t0` and `t1`.
        """
        pass
