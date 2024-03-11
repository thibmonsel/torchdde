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
    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
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
