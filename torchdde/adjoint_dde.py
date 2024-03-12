from typing import Any, Callable, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.integrate import _integrate
from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller.base import AbstractStepSizeController


class nddeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        history_func: Callable[
            [Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]
        ],
        func: torch.nn.Module,
        ts: Float[torch.Tensor, " time"],
        args: Any,
        solver: AbstractOdeSolver,
        stepsize_controller: AbstractStepSizeController,
        *params,  # type: ignore
    ) -> Float[torch.Tensor, "batch time ..."]:
        # Saving parameters for backward()
        ctx.history_func = history_func
        ctx.stepsize_controller = stepsize_controller
        ctx.solver = solver
        ctx.func = func
        ctx.ts = ts

        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys, ys_interpolator = _integrate(
                func,
                solver,
                ts,
                history_func,
                args,
                stepsize_controller,
                delays=func.delays,
            )
        ctx.ys_interpolator = ys_interpolator
        ctx.ys = ys
        ctx.args = args
        return ys

    @staticmethod
    def backward(ctx, *grad_y):  # type: ignore
        # http://www.cs.utoronto.ca/~calver/papers/adjoint_paper.pdf
        # This function implements the adjoint gradient
        # estimation method for NDDEs with constant delays
        # as learnable parameter alongside with the neural network.
        grad_output = grad_y[0]

        args = ctx.args
        T = ctx.ts[-1]
        dt = ctx.ts[1] - ctx.ts[0]
        solver = ctx.solver
        stepsize_controller = ctx.stepsize_controller
        params = ctx.saved_tensors
        state_interpolator = ctx.ys_interpolator

        # computing y'(t-tau) for the contribution of
        # delay parameters in the loss w.r.t to the parameters
        grad_ys = torch.gradient(ctx.ys, dim=1)[0] / dt
        ts_history = torch.linspace(
            ctx.ts[0] - max(ctx.func.delays).item(),
            ctx.ts[0],
            int(max(ctx.func.delays) / dt) + 2,
            device=ctx.ts.device,
        )
        ys_history_eval = torch.concat(
            [torch.unsqueeze(ctx.history_func(t), dim=1) for t in ts_history],
            dim=1,
        )
        if len(ys_history_eval.shape) == 2:
            ys_history_eval = ys_history_eval[..., None]
        grad_ys_history_func = torch.gradient(ys_history_eval, dim=1)[0] / dt
        grad_ys = torch.cat((grad_ys_history_func, grad_ys), dim=1)

        # adjoint history shape [N, N_t=1, D]
        # create an adjoint interpolator that will be
        # used for the integration of the adjoint DDE
        # Our adjoint state is null for t>=T
        adjoint_state = torch.zeros_like(grad_output[:, -1], device=ctx.ts.device)
        adjoint_ys_final = -grad_output[:, -1].reshape(
            adjoint_state.shape[0], 1, *adjoint_state.shape[1:]
        )
        add_t = torch.tensor([ctx.ts[-1], ctx.ts[-1] + dt], device=ctx.ts.device)

        adjoint_interpolator = TorchLinearInterpolator(
            add_t,
            torch.concat([adjoint_ys_final, adjoint_ys_final], dim=1),
        )

        def adjoint_dyn(t, adjoint_y, args):
            h_t = torch.autograd.Variable(
                state_interpolator(t) if t >= ctx.ts[0] else ctx.history_func(t),
                requires_grad=True,
            )
            h_t_minus_tau = [
                (
                    state_interpolator(t - tau)
                    if t - tau >= ctx.ts[0]
                    else ctx.history_func(t - tau)
                )
                for tau in ctx.func.delays
            ]
            out = ctx.func(t, h_t, args, history=h_t_minus_tau)
            # This correspond to the term adjoint(t) df(t, y(t), y(t-tau))_dy(t)
            rhs_adjoint_1 = torch.autograd.grad(
                out,
                h_t,
                -adjoint_y,
                retain_graph=True,
            )[0]

            # we need to add the second term of rhs too in rhs_adjoint computation
            delay_derivative_inc = torch.zeros_like(ctx.func.delays)[..., None]
            for idx, tau_i in enumerate(ctx.func.delays):
                if t < T - tau_i:
                    adjoint_t_plus_tau = adjoint_interpolator(t + tau_i)
                    h_t_plus_tau = state_interpolator(t + tau_i)
                    history = [
                        (
                            state_interpolator(t + tau_i - tau_j)
                            if t + tau_i - tau_j >= ctx.ts[0]
                            else ctx.history_func(t + tau_i - tau_j)
                        )
                        for tau_j in ctx.func.delays
                    ]
                    history[idx] = h_t
                    out_other = ctx.func(t + tau_i, h_t_plus_tau, args, history=history)

                    # This correspond to the term
                    # adjoint(t+tau) df(t+tau, y(t+tau), y(t))_dy(t)
                    rhs_adjoint_2 = torch.autograd.grad(
                        out_other, h_t, -adjoint_t_plus_tau
                    )[0]
                    rhs_adjoint_1 += rhs_adjoint_2

                    # contribution of the delay in the gradient's loss
                    # ie int_0^{T-\tau} \pdv{f(x_{t+\tau}, x_{t})}{x_t} x'(t) dt
                    delay_derivative_inc[idx] += torch.sum(
                        rhs_adjoint_2 * grad_ys[:, -1 - j],
                        dim=(tuple(range(len(rhs_adjoint_2.shape)))),
                    )

            param_derivative_inc = torch.autograd.grad(
                out,
                params,
                -adjoint_y,
                retain_graph=True,
                allow_unused=True,
            )
            return rhs_adjoint_1, (
                param_derivative_inc,
                delay_derivative_inc,
            )

        # computing the adjoint dynamics
        out2, out3, last_out2, last_out3 = None, None, None, None
        delay_derivative_inc = torch.zeros_like(ctx.func.delays)[..., None]

        tnext, controller_state = stepsize_controller.init(
            adjoint_dyn, T, ctx.ts[0], adjoint_state, -dt, args, solver.order
        )
        tprev = T
        for j, current_t in enumerate(reversed(ctx.ts)):
            with torch.enable_grad():
                adjoint_state = adjoint_state - grad_output[:, -j - 1]
                adjoint_interpolator.add_point(tprev, adjoint_state)
                (
                    adj_candidate,
                    adj_error,
                    (param_derivative_inc, delay_derivative_inc),
                ) = solver.step(
                    adjoint_dyn, current_t, adjoint_state, -dt, args, has_aux=True
                )
                (
                    keep_step,
                    tprev,
                    tnext,
                    controller_state,
                ) = stepsize_controller.adapt_step_size(
                    adjoint_dyn,
                    tprev,
                    tnext,
                    adjoint_state,
                    adj_candidate,
                    args,
                    solver.order,
                    adj_error,
                    controller_state,
                )
                if keep_step:
                    adjoint_state = adj_candidate

                    if out2 is None:
                        out2 = tuple([dt / 2 * p for p in param_derivative_inc])
                    else:
                        for _1, _2 in zip([*out2], [*param_derivative_inc]):
                            if _2 is not None:
                                _1 += dt * _2

                    if out3 is None:
                        out3 = tuple([-dt / 2 * p for p in delay_derivative_inc])
                    else:
                        for _1, _2 in zip([*out3], [*delay_derivative_inc]):
                            if _2 is not None:
                                _1 += -dt * _2

                    if current_t == T:
                        last_out2 = tuple([p for p in param_derivative_inc])

        # adding the last contribution of the delay
        # parameters in the loss w.r.t. the parameters
        # ie which is the last part of the integration
        # from t = 0 to t = -tau
        for idx, tau_i in enumerate(ctx.func.delays):
            ts_history_i = torch.linspace(
                ctx.ts[0] - tau_i.item(), ctx.ts[0], int(tau_i.item() / dt)
            ).to(ctx.ts.device)
            for k, t in enumerate(reversed(ts_history_i)):
                with torch.enable_grad():
                    h_t = torch.autograd.Variable(
                        (
                            state_interpolator(t)
                            if t >= ctx.ts[0]
                            else ctx.history_func(t)
                        ),
                        requires_grad=True,
                    )

                    adjoint_t_plus_tau = adjoint_interpolator(t + tau_i)
                    h_t_plus_tau = state_interpolator(t + tau_i)
                    history = [
                        (
                            state_interpolator(t + tau_i - tau_j)
                            if t + tau_i - tau_j >= ctx.ts[0]
                            else ctx.history_func(t + tau_i - tau_j)
                        )
                        for tau_j in ctx.func.delays
                    ]
                    history[idx] = h_t
                    out_other = ctx.func(t + tau_i, h_t_plus_tau, args, history=history)
                    rhs_adjoint_inc = torch.autograd.grad(
                        out_other, h_t, -adjoint_t_plus_tau
                    )[0]

                    # remaining contribution of the delay in the gradient's
                    # loss ie int_{-\tau}^{0} \pdv{f(x_{t+\tau}, x_{t})}{x_t} x'(t) dt
                    delay_derivative_inc[idx] += torch.sum(
                        rhs_adjoint_inc * grad_ys[:, k],
                        dim=(tuple(range(len(rhs_adjoint_inc.shape)))),
                    )

                    if k == len(ts_history_i) - 1:
                        last_out3 = delay_derivative_inc
        if out3 is not None:
            for _1, _2 in zip([*out3], [*delay_derivative_inc]):
                if _2 is not None:
                    _1 += -dt * _2

        # Adding last term in order to get a trapz rule
        # estimate of the grad wtr to the parameters
        # trapezoid is h/2 * (f(a) + f(b) + 2 (f(x1) + ... + f(xn-1)))
        # compared to rectangle rule is h * (f(a) + f(b) + f(x1) + ... + f(xn-1))
        if out2 is not None:
            if last_out2 is not None:
                for _1, _2 in zip([*out2], [*last_out2]):
                    if _2 is not None:
                        _1 += dt / 2 * _2

        if out3 is not None:
            if last_out3 is not None:
                for _1, _2 in zip([*out3], [*last_out3]):
                    if _2 is not None:
                        _1 -= -dt / 2 * _2
        return None, None, None, None, None, None, *(out3[0] + out2[0], *out2[1:])  # type: ignore


def ddesolve_adjoint(
    history_func: Callable[[Float[torch.Tensor, ""]], Float[torch.Tensor, "batch ..."]],
    func: torch.nn.Module,
    ts: Float[torch.Tensor, " time"],
    args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController,
) -> Union[Float[torch.Tensor, "batch time ..."], Any]:
    r"""Main function to integrate a constant time delay DDE with the adjoint method

    **Arguments:**

    - `history_func`: DDE's history function
    - `func`: Pytorch model, i.e vector field
    - `ts`: Integration span
    - `solver`: ODE solver use

    **Returns:**

    Integration result over `ts`.
    """
    params = find_parameters(func)
    ys = nddeint_ACA.apply(
        history_func, func, ts, args, solver, stepsize_controller, *params
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
