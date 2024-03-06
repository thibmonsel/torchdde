from typing import Any, Callable, Union

import torch
from jaxtyping import Float
from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.base import AbstractOdeSolver


def integrate(
    func: Union[torch.nn.Module, Callable],
    solver: AbstractOdeSolver,
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    args: Any,
    delays: Float[torch.Tensor, "batch ..."] = None,
    discretize_then_optimize: bool = False,
):
    # imported here to handle circular dependencies
    # not sure this is the best...
    from torchdde.adjoint_dde import ddesolve_adjoint
    from torchdde.adjoint_ode import odesolve_adjoint

    if discretize_then_optimize or not isinstance(func, torch.nn.Module):
        return _integrate(func, solver, ts, y0, args, delays)
    else:
        if delays is not None:
            return ddesolve_adjoint(y0, func, ts, args, solver), None
        else:
            return odesolve_adjoint(y0, func, ts, args, solver)


def _integrate(
    func: Union[torch.nn.Module, Callable],
    solver: AbstractOdeSolver,
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    args: Any,
    delays: Float[torch.Tensor, "batch ..."] = None,
) -> Float[torch.Tensor, "batch ..."]:
    r"""Integrate a system of ODEs.
    **Arguments:**

    - `func`: Pytorch model, i.e vector field
    - `ts`: Integration span
    - `y0`: Initial condition for ODE / History function for DDE
    - `has_aux`: Whether the model has an auxiliary output.

    **Returns:**

    Integration result over `ts`
    """
    if delays is not None:
        history_func = y0
        y0 = history_func(ts[0])

        return _integrate_dde(func, ts, y0, history_func, args, delays, solver)
    else:
        return _integrate_ode(func, ts, y0, args, solver)


def _integrate_dde(func, ts, y0, history_func, args, delays, solver):

    dt = ts[1] - ts[0]
    # y0 should have the shape [batch, N_t=1, features]
    # in order to properly instantiate the
    # interpolator class
    y0 = torch.unsqueeze(y0.clone(), dim=1)
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
            (ys_interpolation(t - tau) if t - tau >= ts[0] else history_func(t - tau))
            for tau in delays
        ]
        return func(t, y, args, history=history)

    current_y = y0[:, 0]
    ys = torch.unsqueeze(current_y, dim=1)
    current_t = ts[0]
    while current_t < ts[-1]:
        # the stepping method give the next y
        # with a shape [batch, features]
        y, _ = solver.step(ode_func, current_t, ys[:, -1], dt, args)  # type: ignore
        current_t = current_t + dt
        # by adding the y to the interpolator,
        # it is unsqueezed in the interpolator class
        ys_interpolation.add_point(current_t + dt, y)
        ys = torch.concat((ys, torch.unsqueeze(y, dim=1)), dim=1)

    return ys, ys_interpolation


def _integrate_ode(func, ts, y0, args, solver):
    dt = ts[1] - ts[0]
    ys = torch.unsqueeze(y0.clone(), dim=1)
    current_t = ts[0]
    while current_t <= ts[-1]:
        y, _ = solver.step(func, current_t, ys[:, -1], dt, args, has_aux=False)
        current_t = current_t + dt
        ys = torch.cat((ys, torch.unsqueeze(y, dim=1)), dim=1)
    return ys
