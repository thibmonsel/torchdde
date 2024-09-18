from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from jaxtyping import Float, Int

from torchdde.global_interpolation.dense_interpolation import DenseInterpolation
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller import (
    AbstractStepSizeController,
    ConstantStepSizeController,
)


@dataclass
class State:
    """Evolving state during the solve"""

    y: Float[torch.Tensor, "batch ..."]
    tprev: Float[torch.Tensor, ""]
    tnext: Float[torch.Tensor, ""]
    # solver_state: PyTree[ArrayLike]
    dt: Float[torch.Tensor, ""]
    # result: RESULTS
    num_steps: Int[torch.Tensor, " 1"]
    num_accepted_steps: Int[torch.Tensor, " 1"]
    num_rejected_steps: Int[torch.Tensor, " 1"]
    save_idx: Int[torch.Tensor, " 1"]

    def __repr__(self) -> str:
        return f"""State(
            y={self.y},
            tprev={self.tprev}, 
            tnext={self.tnext}, 
            dt={self.dt}, 
            num_step={self.num_steps.item()},
            num_accepted_steps={self.num_accepted_steps.item()}, 
            num_rejected_steps={self.num_rejected_steps.item()}, 
            save_idx={self.save_idx.item()}\n)"""


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
    max_steps: int = 2048,
) -> Float[torch.Tensor, "batch time ..."]:
    """Solves a system of ODEs or DDEs.
    See the [Getting started](../usage/getting-started.md) page for example usage.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `func`: The terms of the differential equation. This specifies the vector field.
    - `solver`: The solver for the differential equation. See the
        available [solvers](../usage/solvers.md).
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `ts` : The time points at which to return the solution.
    - `y0`: The initial value. This is either a tensor (for ODEs) or a
            history function callable (for DDEs)
    - `args`: Any additional arguments to pass to the vector field.
    - `stepsize_controller`: How to change the step size as the integration progresses.
        See the [list of stepsize controllers](../usage/stepsize-controller.md).
        Defaults set to `ConstantStepSizeController`.
    - `dt0`: The step size to use for the first step. If you are
        using a fixed step sizes then this will also be the step size
        for all other steps. If set as `None` then the
        initial step size will be determined automatically,
        only available for `torchdde.AdaptiveStepSizeController`.
    - `delays`: The initial values given to the constant delays for the DDE.
        If set to `None` then the system is treated as an ODE.
    - `discretize_then_optimize` : If set to `False`, the adjoint method
        will be used for training.
        If set to `True`, regular backpropagation will be used for training.
    - `max_steps` : The maximum number of steps to take before
    quitting the computation unconditionally.

    **Returns:**

    Returns the solution of the differential equation.
    """

    # imported here to handle circular dependencies
    # this surely isn't the best...
    from torchdde.adjoint_dde import ddesolve_adjoint
    from torchdde.adjoint_ode import odesolve_adjoint

    if discretize_then_optimize or not isinstance(func, torch.nn.Module):
        return _integrate(
            func,
            solver,
            t0,
            t1,
            ts,
            y0,
            args,
            stepsize_controller,
            dt0,
            delays,
            max_steps,
        )[0]
    else:
        if delays is not None:
            # y0 is a Callable that encaptulates
            # the history function and y0 at the same time
            # by enforcing that history_func = y0 and
            # history_func(ts[0]) = y0 in _integrate
            assert isinstance(y0, Callable)
            return ddesolve_adjoint(
                func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0, max_steps
            )
        else:
            assert isinstance(y0, torch.Tensor)
            return odesolve_adjoint(
                func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0, max_steps
            )


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
    max_steps: int = 100,
    has_aux: bool = False,
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
        y0_ = history_func(t0)
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
            max_steps=max_steps,
            has_aux=has_aux,
        )
    else:
        assert isinstance(y0, torch.Tensor)
        return _integrate_ode(
            func,
            t0,
            t1,
            ts,
            y0,
            args,
            solver,
            stepsize_controller,
            dt0,
            max_steps=max_steps,
            has_aux=has_aux,
        )


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
    max_steps: Optional[int] = 100,
    has_aux: bool = False,
) -> tuple[
    Float[torch.Tensor, "batch time ..."],
    tuple[
        Optional[Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]]],
        Any,
    ],
]:
    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    if dt0 is not None:
        if dt0 * (t1 - t0) < 0:
            raise ValueError("Must have (t1 - t0) * dt0 >= 0")

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
        def cond(t, tau):
            in_history_domain = (t - tau).item() <= t0
            very_close_to_t0 = torch.allclose(t - tau, t0)
            return in_history_domain or very_close_to_t0

        history = [
            history_func(t - tau) if cond(t, tau) else ys_interpolation(t - tau)  # type: ignore
            for tau in delays
        ]
        return func(t, y, args, history=history)

    tnext, dt = stepsize_controller.init(func, t0, t1, y0, dt0, args, solver.order())
    dt = torch.clamp(dt, max=torch.min(delays))

    state = State(
        y0,
        t0,
        tnext,
        dt,
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
    )
    ys = torch.empty(
        (y0.shape[0], ts.shape[0], *(y0.shape[1:])), device=y0.device, dtype=y0.dtype
    )
    ys_interpolation = None
    cond = state.tprev < t1 if (t1 > t0) else state.tprev > t1
    while cond and state.num_steps < max_steps:
        y, y_error, dense_info, aux = solver.step(
            ode_func, state.tprev, state.y, state.dt, args, has_aux=has_aux
        )
        (
            keep_step,
            tprev,
            tnext,
            dt,
        ) = stepsize_controller.adapt_step_size(
            ode_func,
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            solver.order(),
            state.dt,
        )
        tprev = torch.clamp(tprev, max=t1)
        # if the next step going beyond the smallest delay
        # then our ys_interpolation isn't defined
        # TODO: do a fixed point iteration algorithm
        too_large = (tnext - tprev) > torch.min(delays)
        tnext = torch.where(too_large, tprev + torch.min(delays), tnext)
        dt = torch.where(too_large, torch.min(delays), dt)
        step_save_idx = 0
        if keep_step:
            interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
            #### Updating interpolators ####
            if ys_interpolation is None:
                ys_interpolation = DenseInterpolation(
                    ts[1] - ts[0] > 0,
                    solver.interpolation_cls,
                    torch.tensor([t0, state.tnext], device=y.device),
                    dict(
                        zip(
                            dense_info.keys(),
                            [dense_info[key].unsqueeze(0) for key in dense_info.keys()],
                        )
                    ),
                )
            else:
                ys_interpolation.add_point(state.tnext.reshape(1), dense_info)

            while torch.any(state.tnext >= ts[state.save_idx + step_save_idx :]):
                #### Bookkeeping, saving values ####
                idx = state.save_idx + step_save_idx
                out = interp(ts[idx])
                ys[:, idx] = (
                    out.unsqueeze(1) if len(out.shape) != len(ys[:, idx].shape) else out
                )
                step_save_idx += 1

        ########################################
        ##### Updating State for next step #####
        ########################################
        y = torch.where(keep_step, y, state.y)
        num_accepted_steps = torch.where(
            keep_step, state.num_accepted_steps + 1, state.num_accepted_steps
        )
        num_rejected_steps = torch.where(
            keep_step, state.num_rejected_steps, state.num_rejected_steps + 1
        )
        save_idx = torch.where(
            keep_step, state.save_idx + step_save_idx, state.save_idx
        )

        state = State(
            y,
            tprev,
            tnext,
            dt,
            state.num_steps + 1,
            num_accepted_steps,
            num_rejected_steps,
            save_idx,
        )
        cond = tprev < t1 if (t1 > t0) else tprev > t1
    if state.num_steps >= max_steps:
        raise RuntimeError("Maximum number of steps reached")
    return ys, (ys_interpolation, aux)  # type: ignore


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
    max_steps: Optional[int] = 100,
    has_aux: bool = False,
) -> tuple[Float[torch.Tensor, "batch time ..."], Any]:
    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    if dt0 is not None:
        if dt0 * (t1 - t0) < 0:
            raise ValueError("Must have (t1 - t0) * dt0 >= 0")

    tnext, dt = stepsize_controller.init(func, t0, t1, y0, dt0, args, solver.order())

    state = State(
        y0,
        t0,
        tnext,
        dt,
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
        torch.tensor([0], device=y0.device),
    )
    ys = torch.empty(
        (y0.shape[0], ts.shape[0], *(y0.shape[1:])), device=y0.device, dtype=y0.dtype
    )
    cond = state.tprev < t1 if (t1 > t0) else state.tprev > t1
    while cond and state.num_steps < max_steps:
        y, y_error, dense_info, aux = solver.step(
            func, state.tprev, state.y, state.dt, args, has_aux=has_aux
        )
        (
            keep_step,
            tprev,
            tnext,
            dt,
        ) = stepsize_controller.adapt_step_size(
            func,
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            solver.order(),
            state.dt,
        )
        tprev = torch.clamp(tprev, max=t1)
        step_save_idx = 0
        if keep_step:
            interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
            while torch.any(state.tnext >= ts[state.save_idx + step_save_idx :]):
                idx = state.save_idx + step_save_idx
                out = interp(ts[idx])
                ys[:, idx] = (
                    out.unsqueeze(1) if len(out.shape) != len(ys[:, idx].shape) else out
                )
                step_save_idx += 1

        ########################################
        ##### Updating State for next step #####
        ########################################

        y = torch.where(keep_step, y, state.y)
        num_accepted_steps = torch.where(
            keep_step, state.num_accepted_steps + 1, state.num_accepted_steps
        )
        num_rejected_steps = torch.where(
            keep_step, state.num_rejected_steps, state.num_rejected_steps + 1
        )
        save_idx = torch.where(
            keep_step, state.save_idx + step_save_idx, state.save_idx
        )

        state = State(
            y,
            tprev,
            tnext,
            dt,
            state.num_steps + 1,
            num_accepted_steps,
            num_rejected_steps,
            save_idx,
        )

        cond = tprev < t1 if (t1 > t0) else tprev > t1
    if state.num_steps >= max_steps:
        raise RuntimeError(
            f"Maximum number of steps reached \
            with solver (max_steps={state.num_steps} \
            {state.num_accepted_steps} accepted)"
        )
    return ys, aux  # type: ignore
