from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Float


class AbstractStepSizeController(ABC):
    @abstractmethod
    def init(
        self, func, t0, t1, y0, dt0, args, error_order
    ) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        """
        **Returns:**

        A 2-tuple of:

        - The endpoint $t0+dt0$ for the initial first step: the first step will be made
            over the interval $[t_0, t1]$. If `dt0` is specified (not `None`) then
            this is typically `t0 + dt0`. (Although in principle the step size
            controller doesn't have to respect this if it doesn't want to.)
        - The initial hidden state for the step size controller, which is used the
            first time `adapt_step_size` is called.
        """
        pass

    @abstractmethod
    def adapt_step_size(
        self,
        func,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        dt,
    ) -> tuple[
        Bool[torch.Tensor, ""],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        """
        **Returns:**

        A tuple of several objects:

        - A boolean indicating whether the step was accepted/rejected.
        - The time at which the next step is to be started.
        - The time at which the next step is to finish.
        - The value of the step size controller state at `t1`.
        """
        pass
