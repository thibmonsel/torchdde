from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.global_interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.integrate import _evaluate_history_many, _integrate_dde, _integrate_ode
from torchdde.misc import TupleTensorTransformer
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller.base import AbstractStepSizeController
from torchdde.step_size_controller.constant import ConstantStepSizeController


def _evaluate_future_histories(future_times, delays, boundary, lookup):
    """Collect history rows for all active future times in their original order."""
    query_times = (future_times[:, None] - delays[None, :]).reshape(-1)
    values = lookup(query_times, boundary)
    width = delays.numel()
    return [values[start : start + width] for start in range(0, len(values), width)]


def _evaluate_trajectory_inputs(times, delays, boundary, interpolation, lookup):
    """Group states and history rows used by learned-delay derivatives."""
    states = interpolation.evaluate_many(times).unbind(0)
    histories = _evaluate_future_histories(times, delays, boundary, lookup)
    return list(zip(states, histories))


def _adjoint_breakpoints(t0, t1, ts, delays, grad_output):
    """Times where the adjoint dynamics jump, sorted increasingly.

    For each delay tau: the future term switches on at T - tau, each loss
    jump at an output time t_k reaches back to t_k - tau through lambda(t + tau),
    and the trajectory derivative y'(t - tau) jumps where t - tau = t0.
    """
    taus = delays.detach().reshape(-1).tolist()
    has_jump = (grad_output != 0).reshape(
        grad_output.shape[0], grad_output.shape[1], -1
    ).any(dim=2).any(dim=0).tolist()
    jump_times = [t for t, jump in zip(ts.detach().tolist(), has_jump) if jump]
    start, end = float(t0), float(t1)
    points = set()
    for tau in taus:
        points.update((end - tau, start + tau))
        points.update(t - tau for t in jump_times)
    return sorted(points)


def _split_interval(start, end, breakpoints):
    """Nodes from `start` to `end` (backwards) with the breakpoints inside.

    Breakpoints within a tiny fraction of the interval of an endpoint are
    treated as lying on it: that endpoint is already read from the inside.
    """
    hi, lo = float(start), float(end)
    # Below the time dtype's resolution two times are indistinguishable.
    resolution = 64 * torch.finfo(start.dtype).eps * max(1.0, abs(hi), abs(lo))
    tolerance = max(1e-6 * (hi - lo), resolution)
    inside = [b for b in reversed(breakpoints) if lo + tolerance < b < hi - tolerance]
    nodes = [start]
    for value in start.new_tensor(inside).unbind() if inside else ():
        # Merge breakpoints closer than the tolerance (rounding to the time
        # dtype can also make neighbours coincide).
        if float(nodes[-1]) - float(value) > tolerance:
            nodes.append(value)
    if len(nodes) > 1 and float(nodes[-1]) - lo <= tolerance:
        nodes.pop()
    nodes.append(end)
    return nodes


class nddeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        func: torch.nn.Module,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        ts: Float[torch.Tensor, " time"],
        history_func: Callable[
            [Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]
        ],
        func_args: Any,
        solver: AbstractOdeSolver,
        stepsize_controller: AbstractStepSizeController = ConstantStepSizeController(),
        dt0: Optional[Float[torch.Tensor, ""]] = None,
        max_steps: Optional[int] = 2048,
        split_at_breakpoints: bool = True,
        *params,  # type: ignore
    ) -> Float[torch.Tensor, "batch time ..."]:
        # Saving parameters for backward()
        ctx.split_at_breakpoints = split_at_breakpoints
        ctx.history_func = history_func
        ctx.stepsize_controller = stepsize_controller
        ctx.solver = solver
        ctx.func = func
        ctx.func_args = func_args
        ctx.ts = ts
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.max_steps = max_steps
        ctx.delay_param_index = next(
            (i for i, param in enumerate(params) if param is func.delays), None
        )

        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys, (ys_interpolator, _) = _integrate_dde(  # type: ignore
                func,
                t0,
                t1,
                ts,
                history_func(t0),
                history_func,
                func_args,
                func.delays,  # type: ignore
                solver,
                stepsize_controller,
                dt0=dt0,
                max_steps=max_steps,
            )

        ctx.ys_interpolator = ys_interpolator
        ctx.ys = ys
        return ys

    @staticmethod
    def backward(ctx, *grad_y) -> Any:  # type: ignore
        # http://www.cs.utoronto.ca/~calver/papers/adjoint_paper.pdf
        # This function implements the adjoint gradient
        # estimation method for NDDEs with constant delays
        # as learnable parameter alongside with the neural network.
        grad_output = grad_y[0]

        func_args = ctx.func_args
        ts = ctx.ts
        dt = ts[1] - ts[0]
        solver = ctx.solver
        stepsize_controller = ctx.stepsize_controller
        params = ctx.saved_tensors
        # Fixed delays do not need trajectory derivatives for delay gradients.
        compute_delay_gradient = ctx.delay_param_index is not None
        state_interpolator = ctx.ys_interpolator
        history_indices = {}
        future_indices = {}
        trajectory_indices = {}
        batch_history = ctx.func.delays.numel() > 1

        def state_history(times, boundary):
            return _evaluate_history_many(
                times, boundary, ctx.history_func, state_interpolator, history_indices,
            )

        # adjoint history shape [N, N_t=1, D]
        # create an adjoint interpolator that will be
        # used for the integration of the adjoint DDE
        # Our adjoint state is null for t>=T
        adjoint_state = torch.zeros_like(grad_output[:, -1], device=ctx.ts.device)
        adjoint_ys_final = -grad_output[:, -1].reshape(
            adjoint_state.shape[0], 1, *adjoint_state.shape[1:]
        )
        add_t = torch.tensor(
            [ctx.t1, 2 * max(ctx.func.delays) + ctx.t1], device=ctx.ts.device
        )

        adjoint_interpolator = TorchLinearInterpolator(
            add_t,
            torch.concat([adjoint_ys_final, adjoint_ys_final], dim=1),
        )

        # augment_state = [adjoint_state, params_incr]
        aug_state = [torch.zeros_like(adjoint_state)]
        aug_state.extend([torch.zeros_like(param) for param in params])
        transformer = TupleTensorTransformer.from_tuple(aug_state)

        # Bounds of the backward interval being integrated, set in the loop
        # below. Several switches in the adjoint dynamics jump at breakpoints
        # (t = T - tau, t = t_k - tau, t - tau = t0). Runge-Kutta stages sit on
        # both interval endpoints, so a breakpoint on an endpoint must be seen
        # from inside the interval: the stages at its top need the value below,
        # those at its bottom the value above.
        interior_bounds = [None, None]

        def interior_time(t):
            """`t` moved just inside the current interval, to decide switches."""
            return torch.clamp(t, *interior_bounds)

        def trajectory_derivative(t, state=None, on_trajectory=None):
            if on_trajectory is None:
                on_trajectory = bool(t >= ctx.t0)
            if on_trajectory:
                y_t = state_interpolator(t) if state is None else state
                history = (
                    state_history(t - ctx.func.delays, ctx.t0) if batch_history else [
                        state_interpolator(t - tau) if t - tau > ctx.t0
                        else ctx.history_func(t - tau) for tau in ctx.func.delays
                    ]
                )
                with torch.no_grad():
                    return ctx.func(t, y_t, func_args, history=history)
            history_time = t.detach().requires_grad_(True)
            _, history_derivative = torch.autograd.functional.jvp(
                ctx.history_func,
                history_time,
                torch.ones_like(history_time),
                strict=False,
            )
            return history_derivative

        def augment_dyn(t, aug_state, func_args):
            adjoint_y, *params_inc = transformer.unflatten(aug_state)
            y_t = torch.autograd.Variable(
                state_interpolator(t) if t > ctx.t0 else ctx.history_func(t),
                requires_grad=True,
            )
            history_values = (
                state_history(t - ctx.func.delays, ctx.t0) if batch_history else [
                    state_interpolator(t - tau) if t - tau > ctx.t0
                    else ctx.history_func(t - tau) for tau in ctx.func.delays
                ]
            )
            h_t_minus_tau = [
                torch.autograd.Variable(value, requires_grad=True)
                for value in history_values
            ]
            func_t = ctx.func(t, y_t, func_args, history=h_t_minus_tau)
            # This correspond to the both terms :
            # \lambda_t \partial{f_\theta(t)}{g} in adjoint dynamics
            # \lambda_t \partial{f_\theta(t)}{\theta} and in loss equation
            gradients = torch.autograd.grad(
                func_t,
                (y_t, *h_t_minus_tau, *params),
                -adjoint_y,
                allow_unused=True,
            )
            rhs_adjoint_1 = gradients[0]
            history_gradients = gradients[1 : 1 + len(h_t_minus_tau)]
            params_inc = gradients[1 + len(h_t_minus_tau) :]
            if rhs_adjoint_1 is None:
                rhs_adjoint_1 = torch.zeros_like(y_t)

            trajectory_inputs = iter(())
            if compute_delay_gradient and batch_history:
                trajectory_times = t - ctx.func.delays
                trajectory_active = (
                    interior_time(t) - ctx.func.delays >= ctx.t0
                ).tolist()
                positions = tuple(
                    i for i, active in enumerate(trajectory_active)
                    if active and history_gradients[i] is not None
                )
                if positions:
                    if positions not in trajectory_indices:
                        trajectory_indices[positions] = torch.tensor(positions, device=t.device)
                    # A breakpoint read from inside the interval can be active
                    # while t - tau rounds to just below t0: query at t0 then.
                    selected_times = torch.clamp(
                        trajectory_times.index_select(0, trajectory_indices[positions]),
                        min=ctx.t0,
                    )
                    trajectory_inputs = iter(_evaluate_trajectory_inputs(
                        selected_times, ctx.func.delays, ctx.t0, state_interpolator,
                        state_history,
                    ))

            # we need to add the second term of rhs in rhs_adjoint computation
            if compute_delay_gradient:
                delay_derivative_inc = torch.zeros_like(ctx.func.delays)[..., None]
            if batch_history:
                # Preserve the scalar branch condition, including its strict
                # terminal boundary, and fetch only the active future values.
                t_inside = interior_time(t)
                active_future = (t_inside < ctx.t1 - ctx.func.delays).tolist()
                positions = tuple(i for i, active in enumerate(active_future) if active)
                future_values = iter(())
                if positions:
                    if positions not in future_indices:
                        future_indices[positions] = torch.tensor(positions, device=t.device)
                    future_times = (t + ctx.func.delays).index_select(
                        0, future_indices[positions],
                    )
                    # The adjoint jumps at output times: look it up from inside.
                    future_values = iter(adjoint_interpolator.evaluate_many(
                        (t_inside + ctx.func.delays).index_select(
                            0, future_indices[positions],
                        )
                    ).unbind(0))
                    future_states = iter(
                        state_interpolator.evaluate_many(future_times).unbind(0)
                    )
                    future_histories = iter(_evaluate_future_histories(
                        future_times, ctx.func.delays, ctx.ts[0], state_history,
                    ))
            for idx, tau_i in enumerate(ctx.func.delays):
                # This is computing part of the contribution of the gradient's
                # loss w.r.t the parameters
                # \lambda(t) \partial{f_\theta(t)}{g}
                # where g(t) = y(t-tau_i)
                params_inc2 = history_gradients[idx]

                if compute_delay_gradient and params_inc2 is not None:
                    if batch_history and trajectory_active[idx]:
                        state, history = next(trajectory_inputs)
                        with torch.no_grad():
                            derivative = ctx.func(t - tau_i, state, func_args, history=history)
                    else:
                        # y(t - tau_i) was already looked up for this call.
                        reuse = not batch_history and bool(t - tau_i > ctx.t0)
                        on_trajectory = bool(interior_time(t) - tau_i >= ctx.t0)
                        derivative = trajectory_derivative(
                            torch.clamp(t - tau_i, min=ctx.t0) if on_trajectory else t - tau_i,
                            history_values[idx] if reuse else None,
                            on_trajectory=on_trajectory,
                        )
                    delay_derivative_inc[idx] += torch.sum(
                        params_inc2 * derivative,
                        dim=(tuple(range(len(params_inc2.shape)))),
                    )

                # if t+ tau_i > T then \lambda(t+tau_i) = 0
                # computing second term of the adjoint dynamics
                # \lambda_t \partial{f_\theta(t)}{g}
                has_future = (
                    active_future[idx] if batch_history
                    else interior_time(t) < ctx.t1 - tau_i
                )
                if has_future:
                    adjoint_t_plus_tau = (
                        next(future_values) if batch_history
                        else adjoint_interpolator(interior_time(t) + tau_i)
                    )
                    y_t_plus_tau = (
                        next(future_states) if batch_history
                        else state_interpolator(t + tau_i)
                    )
                    history = (
                        next(future_histories)
                        if batch_history else [
                            # Entry idx is y(t) and is replaced by y_t below.
                            None if j == idx
                            else state_interpolator(t + tau_i - tau_j)
                            if t + tau_i - tau_j > ctx.ts[0]
                            else ctx.history_func(t + tau_i - tau_j)
                            for j, tau_j in enumerate(ctx.func.delays)
                        ]
                    )
                    history[idx] = y_t
                    func_t_plus_tau_i = ctx.func(
                        t + tau_i, y_t_plus_tau, func_args, history=history
                    )

                    # This correspond to the term
                    # \lambda(t+tau_i) \partial{f_\theta(t+tau_i)}{y_i}
                    # where y_i(t) = g(t-\tau_i)
                    rhs_adjoint_2 = torch.autograd.grad(
                        func_t_plus_tau_i, y_t, -adjoint_t_plus_tau, allow_unused=True
                    )[0]
                    if rhs_adjoint_2 is not None:
                        rhs_adjoint_1 += rhs_adjoint_2

            params_inc = tuple(
                grad if grad is not None else torch.zeros_like(param)
                for grad, param in zip(params_inc, params)
            )
            if ctx.delay_param_index is not None:
                params_inc = list(params_inc)
                params_inc[ctx.delay_param_index] = (
                    params_inc[ctx.delay_param_index] - delay_derivative_inc.squeeze(1)
                )
                params_inc = tuple(params_inc)

            return transformer.flatten(
                (rhs_adjoint_1, *params_inc)
            )

        def set_interior_bounds(start, end):
            # Nudge by a few ulps of the time scale: enough to absorb rounding
            # in t_k - tau, negligible against the step itself.
            nudge = torch.minimum(
                16 * torch.finfo(start.dtype).eps * torch.clamp(start.abs(), min=1.0),
                (start - end).abs() / 4,
            )
            interior_bounds[:] = [
                torch.minimum(start, end) + nudge, torch.maximum(start, end) - nudge,
            ]

        # Integrating across a jump of the dynamics would cost Runge-Kutta its
        # order, so every breakpoint becomes the endpoint of a sub-interval.
        breakpoints = _adjoint_breakpoints(
            ctx.t0, ctx.t1, ts, ctx.func.delays, grad_output,
        ) if ctx.split_at_breakpoints else []

        # computing the adjoint dynamics
        current_num_steps = 0
        for i in range(len(ts) - 1, 0, -1):
            current_num_steps += 1
            if current_num_steps > ctx.max_steps:
                raise RuntimeError("Maximum number of steps reached")

            t0, t1 = ts[i], ts[i - 1]
            nodes = _split_interval(t0, t1, breakpoints)
            with torch.enable_grad():
                # The loss term at t0 makes the adjoint jump there. Store both
                # one-sided values: lambda(t0^+) (before the jump) starts the
                # interval above t0, lambda(t0^-) serves lookups from below.
                adjoint_right = aug_state[0].clone()
                aug_state[0] += grad_output[:, i]
                adjoint_interpolator.add_point(t0, aug_state[0], right_y=adjoint_right)
                flat_state = transformer.flatten(aug_state)
                for start, end in zip(nodes[:-1], nodes[1:]):
                    set_interior_bounds(start, end)
                    dt = torch.clamp(end - start, max=torch.min(ctx.func.delays))
                    new_aug_state, _ = _integrate_ode(
                        augment_dyn,
                        start,
                        end,
                        end[None],
                        flat_state,
                        func_args,
                        solver,
                        stepsize_controller,
                        dt,
                        ctx.max_steps,
                    )
                    flat_state = new_aug_state[:, 0]
                aug_state = transformer.unflatten(new_aug_state)

        params_incr = aug_state[1:]
        # One None per non-parameter input of forward().
        tuple_nones = (None,) * 11
        return *tuple_nones, *params_incr


def ddesolve_adjoint(
    func: torch.nn.Module,
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    history_func: Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    func_args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController = ConstantStepSizeController(),
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    max_steps: Optional[int] = 2048,
    split_at_breakpoints: bool = True,
) -> Union[Float[torch.Tensor, "batch time ..."], Any]:
    r"""Main function to integrate a constant time delay DDE with the adjoint method

    **Arguments:**

    - `history_func`: DDE's history function
    - `func`: Pytorch model, i.e vector field
    - `ts`: Integration span
    - `solver`: ODE solver use
    - `split_at_breakpoints`: integrate the adjoint piecewise between the
        delay breakpoints (see [`torchdde.integrate`][]).

    **Returns:**

    Integration result over `ts`.
    """
    params = find_parameters(func)
    ys = nddeint_ACA.apply(
        func,
        t0,
        t1,
        ts,
        history_func,
        func_args,
        solver,
        stepsize_controller,
        dt0,
        max_steps,
        split_at_breakpoints,
        *params,
    )
    return ys


def find_parameters(module):
    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return [r for r in module.parameters() if r.requires_grad]
