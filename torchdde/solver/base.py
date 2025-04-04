from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Type, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.base import AbstractLocalInterpolation


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the `init`, `step` and `order` method.
    """

    interpolation_cls: Type[AbstractLocalInterpolation]

    @abstractmethod
    def init(
        self,
        func: Union[torch.nn.Module, Callable],
        t0: Float[torch.Tensor, ""],
        y0: Float[torch.Tensor, "batch ..."],
        dt0: Float[torch.Tensor, ""],
        func_args: Any,  # Renamed from 'args' - holds arguments for 'func'
        *args: Any,  # Captures any *extra* positional args passed to init
        **kwargs: Any,  # Optionally capture extra keyword args too
    ) -> Union[Tuple[Any, ...], None]:
        """
        Initializes the solver state. This method is called before the
        integration starts.

        **Arguments:**

        - `func`: Pytorch model or callable function, i.e vector field
        - `t0`: Start of the integration time `t0`
        - `y0`: Initial state `y0`
        - `dt0`: Initial step size `dt0`
        - `func_args`: Arguments to be passed along to `func` when it's called
                       (e.g., func(t, y, func_args)).
        - `*args`: Additional positional arguments passed to init (use rarely).
        - `**kwargs`: Additional keyword arguments passed to init.

        **Returns**

        The initial solver state, which should be used the first time `step` is called.
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
        solver_state: Union[Tuple[Any, ...], None],
        func_args: Any,
        has_aux: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "batch ..."],
        Union[Float[torch.Tensor, "batch ..."], None],
        dict[str, Float[torch.Tensor, "batch order"]],
        Union[Tuple[Any, ...], None],
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
            def f(t,y,func_args):
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
    def build_interpolation(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "..."]],
    ) -> AbstractLocalInterpolation:
        """Interpolator building method based on the solver used.

        **Arguments:**

        - `t0`: The start of the interval over which the interpolation is defined.
        - `t1`: The end of the interval over which the interpolation is defined.
        - `dense_info`: Dictionary that hold all the information needed to properly
        build the interpolation between `t` and `t+dt`.

        **Returns:**

        A `Callable` that can be used to interpolate the solution between `t0` and `t1`.
        """
        pass
