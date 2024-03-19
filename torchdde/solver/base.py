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
        pass

    @abstractmethod
    def order(self) -> int:
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
        """
        **Returns:**

        A tuple of several objects:

        - The value of the solution at `t+dt`.
        - A local error estimate made during the step. (Used by adaptive step size
            controllers to change the step size.) May be `None` if no estimate was
            made.
        - The value of the solver state at `t+dt`.
        - `has_aux`: Whether the model has an auxiliary output.
        """

        r"""ODE's stepping definition

        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `t`: Current time step t
        - `y`: Current state y
        - `dt`: Stepsize dt
        - `has_aux`: Whether the model has an auxiliary output.

        **Returns:**

        Integration result at time `t+dt`
        """
        pass

    @abstractmethod
    def build_interpolation(self, t0, t1, dense_info) -> Any:
        pass
