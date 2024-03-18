from typing import Any, Callable, Optional, Union

import torch
from jaxtyping import Float

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller import (
    AbstractStepSizeController,
    ConstantStepSizeController,
)


def integrate(
    func: Union[torch.nn.Module, Callable],
    solver: AbstractOdeSolver,
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Union[
        Float[torch.Tensor, "batch ..."],
        Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    ],
    args: Any,
    stepsize_controller: AbstractStepSizeController = ConstantStepSizeController(),
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    delays: Optional[Float[torch.Tensor, " delays"]] = None,
    discretize_then_optimize: bool = False,
) -> Float[torch.Tensor, "batch time ..."]:
    # imported here to handle circular dependencies
    # this surely isn't the best...
    from torchdde.adjoint_dde import ddesolve_adjoint
    from torchdde.adjoint_ode import odesolve_adjoint

    if discretize_then_optimize or not isinstance(func, torch.nn.Module):
        return _integrate(
            func, solver, t0, t1, ts, y0, args, stepsize_controller, dt0, delays
        )[0]
    else:
        if delays is not None:
            # y0 is a Callable that encaptulates
            # the history function and y0 at the same time
            # by enforcing that history_func = y0 and
            # history_func(ts[0]) = y0 in _integrate
            assert isinstance(y0, Callable)
            return ddesolve_adjoint(y0, func, ts, args, solver, stepsize_controller)
        else:
            assert isinstance(y0, torch.Tensor)
            return odesolve_adjoint(y0, func, ts, args, solver, stepsize_controller)


def _integrate(
    func: Union[torch.nn.Module, Callable],
    solver: AbstractOdeSolver,
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Union[
        Float[torch.Tensor, "batch ..."],
        Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    ],
    args: Any,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    delays: Optional[Float[torch.Tensor, " delays"]] = None,
) -> tuple[
    Float[torch.Tensor, "batch time ..."],
    Union[Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]], Any],
]:
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
        assert isinstance(y0, Callable)
        history_func = y0
        y0_ = history_func(ts[0])
        return _integrate_dde(
            func,
            t0,
            t1,
            ts,
            y0_,
            history_func,
            args,
            delays,
            solver,
            stepsize_controller,
            dt0,
        )
    else:
        assert isinstance(y0, torch.Tensor)
        return _integrate_ode(
            func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0
        ), None


def _integrate_dde(
    func: Union[torch.nn.Module, Callable],
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    history_func: Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    args: Any,
    delays: Float[torch.Tensor, " delays"],
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
) -> tuple[
    Float[torch.Tensor, "batch time ..."],
    Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
]:
    # y0 should have the shape [batch, N_t=1, features]
    # in order to properly instantiate the
    # interpolator class
    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    tnext, dt = stepsize_controller.init(func, t0, t1, y0, dt0, args, solver.order())

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
    tprev = ts[0]
    while tprev < ts[-1]:
        # the stepping method give the next y
        # with a shape [batch, features]
        y_candidate, y_error, _ = solver.step(
            ode_func, tprev, ys[:, -1], dt, args, has_aux=False
        )
        keep_step, tprev, tnext, dt = stepsize_controller.adapt_step_size(
            ode_func,
            tprev,
            tnext,
            ys[:, -1],
            y_candidate,
            args,
            solver.order,
            y_error,
            dt,
        )
        # by adding the y to the interpolator,
        # it is unsqueezed in the interpolator class
        y = y_candidate if keep_step else ys[:, -1]
        if keep_step:
            ys_interpolation.add_point(tprev, y)
            ys = torch.concat((ys, torch.unsqueeze(y, dim=1)), dim=1)

    return ys, ys_interpolation


def _integrate_ode(
    func: Union[torch.nn.Module, Callable],
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
) -> Float[torch.Tensor, "batch time ..."]:
    tnext, controller_state = stepsize_controller.init(
        func, t0, t1, y0, dt0, args, solver.order
    )
    ys = torch.unsqueeze(y0.clone(), dim=1)
    tprev = ts[0]
    while tprev < ts[-1]:
        y_candidate, y_error, _ = solver.step(
            func, tprev, ys[:, -1], controller_state, args, has_aux=False
        )
        keep_step, tprev, tnext, controller_state = stepsize_controller.adapt_step_size(
            func,
            tprev,
            tnext,
            ys[:, -1],
            y_candidate,
            args,
            solver.order,
            y_error,
            controller_state,
        )
        y = y_candidate if keep_step else ys[:, -1]
        ys = torch.cat((ys, torch.unsqueeze(y, dim=1)), dim=1)
    return ys
