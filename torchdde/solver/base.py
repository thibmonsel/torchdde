from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import torch
from jaxtyping import Float


## Great ressource : https://handwiki.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Ralston's_method


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the step method.
    """

    @abstractmethod
    def init(self):
        """
        Initialize the solver. This method is called before the integration starts.
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """
        Return the order of the solver.
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
        """ODE's stepping method

        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `t`: Current time step `t`
        - `y`: Current state `y`
        - `dt`: Step size `dt`
        - `has_aux`: Whether the model has an auxiliary output.
        **Returns:**

        A tuple of several objects:

        - The value of the solution at `t+dt`.
        - A local error estimate made during the step. (Used by
        `diffrax.AdaptiveStepSizeController` controllers to change the step size.)
        It may be `None` for constant stepsize solver for example.
        - Dictionary that hold all the information needed to properly
        build the interpolation between `t` and `t+dt`.
        - None if the model doesn't have an auxiliary output.
        """
        pass

    @abstractmethod
    def build_interpolation(self, t0, t1, dense_info) -> Any:
        pass
