from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

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
    dt: Float[torch.Tensor, ""]
    solver_state: Union[Tuple[Any, ...], None]
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
    func_args: Any,
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
    - `func_args`: Any additional arguments to pass to the vector field.
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

    if t0 == t1:
        if not torch.all(ts == t0):
            raise ValueError("For a zero-duration solve, all requested times must equal t0.")
        initial = y0(t0) if callable(y0) else y0
        return initial.unsqueeze(1).expand(initial.shape[0], ts.numel(), *initial.shape[1:])

    if delays is not None and t1 < t0:
        raise ValueError("Reverse DDE integration requires future values, not a past history.")

    if discretize_then_optimize or not isinstance(func, torch.nn.Module):
        return _integrate(
            func,
            solver,
            t0,
            t1,
            ts,
            y0,
            func_args,
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
                func,
                t0,
                t1,
                ts,
                y0,
                func_args,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
            )
        else:
            assert isinstance(y0, torch.Tensor)
            return odesolve_adjoint(
                func,
                t0,
                t1,
                ts,
                y0,
                func_args,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
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
    func_args: Any,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    delays: Optional[Float[torch.Tensor, " delays"]] = None,
    max_steps: int = 100,
) -> tuple[
    Float[torch.Tensor, "batch time ..."],
    Union[Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]], Any],
]:
    r"""Integrate a system of ODEs.
    **Arguments:**

    - `func`: Pytorch model, i.e vector field
    - `ts`: Integration span
    - `y0`: Initial condition for ODE / History function for DDE

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
            func_args,
            delays,
            solver,
            stepsize_controller,
            dt0,
            max_steps=max_steps,
        )
    else:
        assert isinstance(y0, torch.Tensor)
        return _integrate_ode(
            func,
            t0,
            t1,
            ts,
            y0,
            func_args,
            solver,
            stepsize_controller,
            dt0,
            max_steps=max_steps,
        )


def _integrate_fixed(func, solver, t0, t1, ts, y0, func_args, dt0, max_steps,
                     append_dense=None):
    """Constant-step solve with host bookkeeping and differentiable state work.

    Callers only select this path when integration times are not trainable.
    DDE callers also require dt0 strictly below every delay, leaving delay-cap
    boundaries on the general path. State, history and model gradients remain
    on-device and attached to autograd.
    """
    clock, end, clock_dt = torch.stack((t0, t1, dt0)).detach().cpu().unbind()
    save_times = ts.detach().cpu().tolist()
    forward = bool(end > clock)
    if clock == end:
        return y0.unsqueeze(1).expand(y0.shape[0], len(save_times), *y0.shape[1:]), None
    if not torch.isfinite(clock_dt) or clock_dt == 0:
        raise ValueError("Fixed step size must be finite and nonzero")
    if (end - clock) * clock_dt < 0:
        raise ValueError("Must have (t1 - t0) * dt0 >= 0")

    tprev, dt, y = t0, dt0, y0
    ys = y0.new_empty((y0.shape[0], len(save_times), *y0.shape[1:]))
    save_idx, count, aux = 0, 0, None
    solver_state = None
    while bool(clock < end if forward else clock > end):
        if count >= max_steps:
            raise RuntimeError("Maximum number of steps reached")
        next_clock = clock + clock_dt
        clipped = bool(next_clock > end if forward else next_clock < end)
        if clipped:
            next_clock = end
        tnext = t1 if clipped else tprev + dt
        # The general ODE path reconstructs dt from its endpoints every step;
        # DDEs keep the constant dt until the final clipped step.
        if append_dense is None or clipped:
            dt = tnext - tprev
            clock_dt = next_clock - clock
        if count == 0:
            solver_state = solver.init(func, tprev, y, dt, func_args)
        y, _, dense_info, solver_state, aux = solver.step(
            func, tprev, y, dt, solver_state, func_args,
        )
        next_value = float(next_clock)
        needs_output = save_idx < len(save_times) and (
            save_times[save_idx] <= next_value if forward
            else save_times[save_idx] >= next_value
        )
        if append_dense is not None or needs_output:
            interpolation = solver.build_interpolation(tprev, tnext, dense_info)
            if append_dense is not None:
                append_dense(tprev, tnext, dense_info, interpolation)
            while save_idx < len(save_times) and (
                save_times[save_idx] <= next_value if forward
                else save_times[save_idx] >= next_value
            ):
                ys[:, save_idx] = interpolation(ts[save_idx])
                save_idx += 1
        tprev, clock = tnext, next_clock
        count += 1
    if save_idx != len(save_times):
        raise ValueError("Requested times must lie inside the integration interval")
    return ys, aux


def _integrate_dde(
    func: Union[torch.nn.Module, Callable],
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    history_func: Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    func_args: Any,
    delays: Float[torch.Tensor, " delays"],
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    max_steps: Optional[int] = 100,
) -> tuple[
    Float[torch.Tensor, "batch time ..."],
    tuple[
        Optional[Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]]],
        Any,
    ],
]:
    assert max_steps is not None

    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    if dt0 is not None:
        if dt0 * (t1 - t0) < 0:
            raise ValueError("Must have (t1 - t0) * dt0 >= 0")

    # FSAL solvers and adaptive initial-step selection call ode_func during init.
    ys_interpolation = None

    def ode_func(
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        func_args: Any,
    ):
        history = []
        for tau in delays:
            delayed_t = t - tau
            if ys_interpolation is None:
                history.append(history_func(delayed_t))
                continue
            in_history_domain = delayed_t.item() <= t0
            close_to_boundary = torch.allclose(delayed_t, t0)
            history.append(
                history_func(delayed_t)
                if in_history_domain or close_to_boundary
                else ys_interpolation(delayed_t)
            )
        return func(t, y, func_args, history=history)

    def append_dense(tprev, tnext, dense_info, interpolation):
        nonlocal ys_interpolation
        if ys_interpolation is None:
            # The original initial dense times are detached. Do not reuse
            # coefficients with trainable time endpoints in that case.
            shared = interpolation if not (tprev.requires_grad or tnext.requires_grad) else None
            ys_interpolation = DenseInterpolation(
                True,
                solver.interpolation_cls,
                torch.stack((t0, tnext)).detach(),
                {key: value.unsqueeze(0) for key, value in dense_info.items()},
                interpolation=shared,
            )
            ys_interpolation._share_step_interpolation = shared is not None
        else:
            shared = interpolation if ys_interpolation._share_step_interpolation else None
            ys_interpolation.add_point(tnext.reshape(1), dense_info, interpolation=shared)

    if (
        type(stepsize_controller) is ConstantStepSizeController
        and dt0 is not None
        and not any(value.requires_grad for value in (t0, t1, ts, dt0))
        and bool(torch.all(delays > dt0))
    ):
        ys, aux = _integrate_fixed(
            ode_func, solver, t0, t1, ts, y0, func_args, dt0, max_steps, append_dense,
        )
        return ys, (ys_interpolation, aux)

    tnext, dt = stepsize_controller.init(
        ode_func, t0, t1, y0, dt0, func_args, solver.order()
    )
    dt = torch.clamp(dt, max=torch.min(delays))
    solver_state = solver.init(ode_func, t0, y0, dt, func_args)
    state = State(
        y0,
        t0,
        tnext,
        dt,
        solver_state,
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
        y, y_error, dense_info, solver_state, aux = solver.step(
            ode_func,
            state.tprev,
            state.y,
            state.dt,
            state.solver_state,
            func_args,
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
            func_args,
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
        # Clipping tnext if it goes beyond t1
        # and update dt such that tprev + dt = tnext
        tnext = torch.where((tprev + dt) > t1, t1, tprev + dt)
        dt = torch.where((tprev + dt) > t1, t1 - tprev, dt)
        dt = torch.where(too_large, torch.min(delays), dt)
        step_save_idx = 0
        if keep_step:
            interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
            append_dense(state.tprev, state.tnext, dense_info, interp)

            while torch.any(state.tnext >= ts[state.save_idx + step_save_idx :]):
                #### Bookkeeping, saving values ####
                idx = state.save_idx + step_save_idx
                out = interp(ts[idx])
                ys[:, idx] = out.unsqueeze(1) if out.ndim != ys[:, idx].ndim else out
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

        if not keep_step:
            # equivalent to torch.where but solver_state isn't a tensor
            solver_state = state.solver_state

        state = State(
            y,
            tprev,
            tnext,
            dt,
            solver_state,
            state.num_steps + 1,
            num_accepted_steps,
            num_rejected_steps,
            save_idx,
        )
        cond = tprev < t1 if (t1 > t0) else tprev > t1
    if cond:
        raise RuntimeError("Maximum number of steps reached")
    return ys, (ys_interpolation, aux)  # type: ignore


def _integrate_ode(
    func: Union[torch.nn.Module, Callable],
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    func_args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController,
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    max_steps: Optional[int] = 100,
) -> tuple[Float[torch.Tensor, "batch time ..."], Any]:
    assert max_steps is not None

    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    if dt0 is not None:
        if dt0 * (t1 - t0) < 0:
            raise ValueError("Must have (t1 - t0) * dt0 >= 0")

    if (
        type(stepsize_controller) is ConstantStepSizeController
        and dt0 is not None
        and not any(value.requires_grad for value in (t0, t1, ts, dt0))
    ):
        return _integrate_fixed(
            func, solver, t0, t1, ts, y0, func_args, dt0, max_steps,
        )

    tnext, dt = stepsize_controller.init(
        func, t0, t1, y0, dt0, func_args, solver.order()
    )
    forward = t1 > t0
    tnext = torch.minimum(tnext, t1) if forward else torch.maximum(tnext, t1)
    dt = tnext - t0
    solver_state = solver.init(func, t0, y0, dt, func_args)
    state = State(
        y0,
        t0,
        tnext,
        dt,
        solver_state,
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
        y, y_error, dense_info, solver_state, aux = solver.step(
            func,
            state.tprev,
            state.y,
            state.dt,
            state.solver_state,
            func_args,
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
            func_args,
            y_error,
            solver.order(),
            state.dt,
        )
        tprev = torch.minimum(tprev, t1) if forward else torch.maximum(tprev, t1)
        # Clipping tnext if it goes beyond t1
        # and update dt such that tprev + dt = tnext
        tnext = torch.minimum(tprev + dt, t1) if forward else torch.maximum(tprev + dt, t1)
        dt = tnext - tprev
        step_save_idx = 0
        if keep_step:
            interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
            while torch.any(
                state.tnext >= ts[state.save_idx + step_save_idx :]
                if forward
                else state.tnext <= ts[state.save_idx + step_save_idx :]
            ):
                idx = state.save_idx + step_save_idx
                out = interp(ts[idx])
                ys[:, idx] = out.unsqueeze(1) if out.ndim != ys[:, idx].ndim else out
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
        if not keep_step:
            # equivalent to torch.where but solver_state isn't a tensor
            solver_state = state.solver_state

        state = State(
            y,
            tprev,
            tnext,
            dt,
            solver_state,
            state.num_steps + 1,
            num_accepted_steps,
            num_rejected_steps,
            save_idx,
        )
        cond = state.tprev < t1 if (t1 > t0) else state.tprev > t1

    if cond:
        raise RuntimeError(
            f"Maximum number of steps reached \
            with solver (max_steps={state.num_steps} \
            {state.num_accepted_steps} accepted)"
        )
    return ys, aux  # type: ignore
