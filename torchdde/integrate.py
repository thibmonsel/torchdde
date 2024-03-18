from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from jaxtyping import Float

from torchdde.global_interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller import (
    AbstractStepSizeController,
    ConstantStepSizeController,
)

from torchdde.solver import Dopri5
@dataclass
class State:
    """Evolving state during the solve"""

    y: Float[torch.Tensor, "batch ..."]
    tprev: Float[torch.Tensor, ""]
    tnext: Float[torch.Tensor, ""]
    # solver_state: PyTree[ArrayLike]
    dt: Float[torch.Tensor, ""]
    # result: RESULTS
    num_steps: int
    num_accepted_steps: int
    num_rejected_steps: int
    save_idx : int

    def __repr__(self) -> str:
        return f"State(\ny={self.y},\n tprev={self.tprev}, \n tnext={self.tnext}, \n dt={self.dt}, \n num_steps={self.num_steps}, \n num_accepted_steps={self.num_accepted_steps}, \n num_rejected_steps={self.num_rejected_steps}\n, save_idx={self.save_idx}\n)"
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
            return odesolve_adjoint(
                func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0
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
        return (
            _integrate_ode(
                func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0
            ),
            None,
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

    tnext, dt = stepsize_controller.init(
        func, t0, t1, y0, dt0, args, solver.order()
    )

    state = State(y0, t0, tnext, dt, 0, 0, 0)
    ys = torch.empty((y0.shape[0], ts.shape[0], *(y0.shape[1:])))
    ys_interpolation = TorchLinearInterpolator(ts[0].view(1), torch.unsqueeze(y0.clone(), dim=1))
    print('ys_interpolation.ts',ys_interpolation.ts)

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

    cond = state.tprev < t1 if (t1 > t0) else state.tprev > t1
    while cond:
        print(state)
        print('ys_interpolation.ts',ys_interpolation.ts)
        y, y_error, dense_info, _ = solver.step(
            ode_func, state.tprev, state.y, state.dt, args, has_aux=False
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

        if keep_step:
            start_idx, end_idx, _ts, _ys = _save(solver, state, ts, dense_info)
            for i, _t in enumerate(_ts) : 
                ys_interpolation.add_point(_t, ys[:, i])

            if end_idx is None:
                ys[:, start_idx:] = _ys
            else:
                ys[:, start_idx:end_idx] = _ys

        tprev = torch.clamp(tprev, max=t1)
        if keep_step:
            state = State(
                y,
                tprev,
                tnext,
                dt,
                state.num_steps + 1,
                state.num_accepted_steps + 1,
                state.num_rejected_steps,
            )
        else:
            state = State(
                state.y,
                tprev,
                tnext,
                dt,
                state.num_steps + 1,
                state.num_accepted_steps,
                state.num_rejected_steps + 1,
            )

        cond = tprev < t1 if (t1 > t0) else tprev > t1
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
    if dt0 is None and isinstance(stepsize_controller, ConstantStepSizeController):
        raise ValueError(
            "Please give a value to dt0 since the stepsize"
            "controller {} cannot have a dt0=None".format(stepsize_controller)
        )

    if dt0 is not None:
        if dt0 * (t1 - t0) < 0:
            raise ValueError("Must have (t1 - t0) * dt0 >= 0")

    tnext, dt = stepsize_controller.init(
        func, t0, t1, y0, dt0, args, solver.order()
    )

    state = State(y0, t0, tnext, dt, 0, 0, 0, 0)
    ys = torch.empty((y0.shape[0], ts.shape[0], *(y0.shape[1:])))
    cond = state.tprev < t1 if (t1 > t0) else state.tprev > t1
    while cond:
        y, y_error, dense_info, _ = solver.step(
            func, state.tprev, state.y, state.dt, args, has_aux=False
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
        
        if keep_step:
            step_save_idx = 0 
            interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
            while torch.any(state.tnext >= ts[state.save_idx + step_save_idx:]) :
                idx = state.save_idx + step_save_idx 
                ys[:, idx] = interp.evaluate(ts[idx]).unsqueeze(1)
                step_save_idx += 1

        if keep_step:
            state = State(
                y,
                tprev,
                tnext,
                dt,
                state.num_steps + 1,
                state.num_accepted_steps + 1,
                state.num_rejected_steps,
                state.save_idx + step_save_idx
            )
        else:
            state = State(
                state.y,
                tprev,
                tnext,
                dt,
                state.num_steps + 1,
                state.num_accepted_steps,
                state.num_rejected_steps + 1,
                state.save_idx
            )

        cond = tprev < t1 if (t1 > t0) else tprev > t1
   
    return ys


def _save(solver, state, ts, dense_info):
    interp = solver.build_interpolation(state.tprev, state.tnext, dense_info)
    if len(ts) == 1:
        new_ys = interp.evaluate(ts)
        new_ys = (
            new_ys.unsqueeze(dim=1)
            if len(new_ys.size()) == len(dense_info["y0"].size())
            else new_ys
        )
        return 0, None, ts, new_ys
    else:
        start_idx = torch.where(ts > state.tprev)[0][0] - 1
        if state.tnext >= ts[-1]:
            ts_sub = ts[start_idx:]
            ts_sub = ts_sub[None, ...] if len(ts_sub.size()) == 0 else ts_sub
            new_ys = interp.evaluate(ts_sub)
            new_ys = (
                new_ys.unsqueeze(dim=1)
                if len(new_ys.size()) == len(dense_info["y0"].size())
                else new_ys
            )
            return start_idx, None, ts_sub, new_ys
        else:
            end_idx = torch.where(ts > state.tnext)[0][0]
            if end_idx != start_idx:
                ts_sub = ts[start_idx:end_idx]
                ts_sub = ts_sub[None, ...] if len(ts_sub.size()) == 0 else ts_sub
                new_ys = interp.evaluate(ts_sub)
                new_ys = (
                    new_ys.unsqueeze(dim=1)
                    if len(new_ys.size()) == len(dense_info["y0"].size())
                    else new_ys
                )
                return start_idx, end_idx, ts_sub, new_ys
