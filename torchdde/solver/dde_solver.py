from typing import Any, Callable, Tuple, Union

import torch
from jaxtyping import Float

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.ode_solver import AbstractOdeSolver


class DDESolver:
    """Solver class used to integrate a DDE with a given ODE solver.
    See [`torchdde.AbstractOdeSolver`][] for more
    details on which solvers are available.
    """

    def __init__(
        self, solver: AbstractOdeSolver, delays: Float[torch.Tensor, " delays"]
    ):
        if torch.min(delays) <= 0:
            raise ValueError("Delays must be positive")
        self.solver = solver
        self.delays = delays

    def integrate(
        self,
        func: Union[torch.nn.Module, Callable],
        ts: Float[torch.Tensor, " time"],
        history_func: Callable,
        args: Any,
    ) -> Tuple[Float[torch.Tensor, "batch time ..."], Callable]:
        r"""Integrate a system of DDEs.

        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `ts`: Integration span
        - `history_func`: DDE's history function

        **Returns:**

        Integration result over `ts` and a `TorchLinearInterpolator`
        object of the result integration.
        """
        dt = ts[1] - ts[0]
        # y0 should have the shape [batch, N_t=1, features]
        # in order to properly instantiate the
        # interpolator class
        y0 = torch.unsqueeze(history_func(ts[0]).clone(), dim=1)
        ys_interpolation = TorchLinearInterpolator(ts[0].view(1), y0)

        def ode_func(
            t: Float[torch.Tensor, ""],
            y: Float[torch.Tensor, "batch ..."],
            args: Any,
        ):
            # applies the function func to the current
            # time t and state y and the history
            # we have to make sur that t - tau > dt
            # otherwise we are making a prediction with
            # an unknown ys_interpolation ...
            history = [
                ys_interpolation(t - tau) if t - tau >= ts[0] else history_func(t - tau)
                for tau in self.delays
            ]
            return func(t, y, args, history=history)

        current_y = y0[:, 0]
        ys = torch.unsqueeze(current_y, dim=1)
        for current_t in ts[:-1]:
            # the stepping method give the next y
            # with a shape [batch, features]
            y, _ = self.solver.step(ode_func, current_t, ys[:, -1], dt, args)  # type: ignore
            # by adding the y to the interpolator,
            # it is unsqueezed in the interpolator class
            ys_interpolation.add_point(current_t + dt, y)
            ys = torch.concat((ys, torch.unsqueeze(y, dim=1)), dim=1)

        return ys, ys_interpolation


DDESolver.__init__.__doc__ = """**Arguments:**

- `solver`: Solver to integrate the DDE
- `delays`: Delays tensors used in DDE

"""
