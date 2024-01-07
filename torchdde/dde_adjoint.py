import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.dde_solver import DDESolver
from torchdde.solver.ode_solver import *


class nddeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, history_func, func, ts, solver, *params):
        # Saving parameters for backward()
        ctx.history_func = history_func
        ctx.solver = solver
        ctx.func = func
        ctx.ts = ts
        with torch.no_grad():
            ctx.save_for_backward(*params)

            # Simulation
            dde_solver = DDESolver(solver, func.delays)
            ys, ys_interpolator = dde_solver.integrate(func, ts, history_func)

        ctx.ys_interpolator = ys_interpolator
        ctx.ys = ys
        return ys

    @staticmethod
    def backward(ctx, *grad_y):
        # http://www.cs.utoronto.ca/~calver/papers/adjoint_paper.pdf
        # This function implements the adjoint gradient estimation method for NDDEs with constant delays
        # as learnable parameter alongside with the neural network.
        grad_output = grad_y[0]

        T = ctx.ts[-1]
        dt = ctx.ts[1] - ctx.ts[0]
        solver = ctx.solver
        params = ctx.saved_tensors
        state_interpolator = ctx.ys_interpolator

        # computing y'(t-tau) for the contribution of delay parameters in the loss w.r.t to the parameters
        grad_ys = torch.gradient(ctx.ys, dim=1)[0] / dt
        ts_history = torch.linspace(
            ctx.ts[0] - max(ctx.func.delays).item(),
            ctx.ts[0],
            int(max(ctx.func.delays) / dt) + 2,
        )
        ts_history = ts_history.to(ctx.ts.device)
        ys_history_eval = torch.concat(
            [torch.unsqueeze(ctx.history_func(t), dim=1) for t in ts_history],
            dim=1,
        )
        if len(ys_history_eval.shape) == 2:
            ys_history_eval = ys_history_eval[..., None]
        grad_ys_history_func = torch.gradient(ys_history_eval, dim=1)[0] / dt
        grad_ys = torch.cat((grad_ys_history_func, grad_ys), dim=1)

        # adjoint history shape [N, N_t=1, D]
        # create an adjoint interpolator that will be used for the integration of the adjoint DDE
        # Our adjoint state is null for t>=T
        adjoint_state = torch.zeros_like(grad_output[:, -1])
        adjoint_ys_final = -grad_output[:, -1].reshape(
            adjoint_state.shape[0], 1, *adjoint_state.shape[1:]
        )
        adjoint_ys_final = adjoint_ys_final.to(ctx.ts.device)
        add_t = torch.tensor([ctx.ts[-1], ctx.ts[-1] + dt])
        add_t = add_t.to(ctx.ts.device)

        adjoint_interpolator = TorchLinearInterpolator(
            add_t,
            torch.concat([adjoint_ys_final, adjoint_ys_final], dim=1),
        )

        def adjoint_dyn(t, adjoint_y, has_aux=False):
            h_t = torch.autograd.Variable(
                state_interpolator(t) if t >= ctx.ts[0] else ctx.history_func(t),
                requires_grad=True,
            )
            h_t_minus_tau = [
                state_interpolator(t - tau)
                if t - tau >= ctx.ts[0]
                else ctx.history_func(t - tau)
                for tau in ctx.func.delays
            ]
            out = ctx.func(t, h_t, history=h_t_minus_tau)
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
                        state_interpolator(t + tau_i - tau_j)
                        if t + tau_i - tau_j >= ctx.ts[0]
                        else ctx.history_func(t + tau_i - tau_j)
                        for tau_j in ctx.func.delays
                    ]
                    history[idx] = h_t
                    out_other = ctx.func(t + tau_i, h_t_plus_tau, history=history)

                    # This correspond to the term adjoint(t+tau) df(t+tau, y(t+tau), y(t))_dy(t)
                    rhs_adjoint_2 = torch.autograd.grad(
                        out_other, h_t, -adjoint_t_plus_tau
                    )[0]
                    rhs_adjoint_1 += rhs_adjoint_2
                    if has_aux:
                        # contribution of the delay in the gradient's loss ie int_0^{T-\tau} \pdv{f(x_{t+\tau}, x_{t})}{x_t} x'(t) dt
                        delay_derivative_inc[idx] += torch.sum(
                            rhs_adjoint_2 * grad_ys[:, -1 - j],
                            dim=(tuple(range(len(rhs_adjoint_2.shape)))),
                        )

            if has_aux:
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
            else:
                return rhs_adjoint_1

        # computing the adjoint dynamics
        out2, out3, last_out2, last_out3 = None, None, None, None
        for j, current_t in enumerate(reversed(ctx.ts)):
            with torch.enable_grad():
                adjoint_state -= grad_output[:, -j - 1]
                adjoint_interpolator.add_point(current_t, adjoint_state)
                adj, (param_derivative_inc, delay_derivative_inc) = solver.step(
                    adjoint_dyn, current_t, adjoint_state, -dt, has_aux=True
                )
                adjoint_state = adj

                if out2 is None:
                    out2 = tuple([dt / 2 * p for p in param_derivative_inc])
                else:
                    for _1, _2 in zip([*out2], [*param_derivative_inc]):
                        _1 += dt * _2 if _2 is not None else 0.0

                if out3 is None:
                    out3 = tuple([-dt / 2 * p for p in delay_derivative_inc])
                else:
                    for _1, _2 in zip([*out3], [*delay_derivative_inc]):
                        _1 += -dt * _2 if _2 is not None else 0.0

                if current_t == T:
                    last_out2 = tuple([p for p in param_derivative_inc])

        # adding the last contribution of the delay parameters in the loss w.r.t. the parameters
        # ie which is the last part of the integration from t = 0 to t = -tau
        for idx, tau_i in enumerate(ctx.func.delays):
            ts_history_i = torch.linspace(
                ctx.ts[0] - tau_i.item(), ctx.ts[0], int(tau_i.item() / dt)
            ).to(ctx.ts.device)
            for k, t in enumerate(reversed(ts_history_i)):
                with torch.enable_grad():
                    h_t = torch.autograd.Variable(
                        state_interpolator(t)
                        if t >= ctx.ts[0]
                        else ctx.history_func(t),
                        requires_grad=True,
                    )
                    # print("adjoint_interpolator.ts", adjoint_interpolator.ts, t, tau_i)
                    adjoint_t_plus_tau = adjoint_interpolator(t + tau_i)
                    h_t_plus_tau = state_interpolator(t + tau_i)
                    history = [
                        state_interpolator(t + tau_i - tau_j)
                        if t + tau_i - tau_j >= ctx.ts[0]
                        else ctx.history_func(t + tau_i - tau_j)
                        for tau_j in ctx.func.delays
                    ]
                    history[idx] = h_t
                    out_other = ctx.func(t + tau_i, h_t_plus_tau, history=history)
                    rhs_adjoint_inc = torch.autograd.grad(
                        out_other, h_t, -adjoint_t_plus_tau
                    )[0]

                    # remaining contribution of the delay in the gradient's loss ie int_{-\tau}^{0} \pdv{f(x_{t+\tau}, x_{t})}{x_t} x'(t) dt
                    delay_derivative_inc[idx] += torch.sum(
                        rhs_adjoint_inc * grad_ys[:, k],
                        dim=(tuple(range(len(rhs_adjoint_inc.shape)))),
                    )

                    if k == len(ts_history_i) - 1:
                        last_out3 = delay_derivative_inc

        for _1, _2 in zip([*out3], [*delay_derivative_inc]):
            _1 += -dt * _2 if _2 is not None else 0.0

        # Adding last term in order to get a trapz rule estimate of the grad wtr to the parameters
        # trapezoid is h/2 * (f(a) + f(b) +2 (f(x1) + ... + f(xn-1)))
        # compared to rectangle rule is h * (f(a) + f(b) + f(x1) + ... + f(xn-1))
        for _1, _2 in zip([*out2], [*last_out2]):
            _1 -= dt / 2 * _2 if _2 is not None else 0.0

        if last_out3 is not None:
            for _1, _2 in zip([*out3], [*last_out3]):
                _1 -= -dt / 2 * _2 if _2 is not None else 0.0

        return None, None, None, None, *(out3[0] + out2[0], *out2[1:])


def ddesolve_adjoint(history, func, ts, solver):
    """Main function to integrate a constant time delay DDE with the adjoint method

    Args:
        history (callable): correspond to the history function of the DDE
        func (nn.Module): neural network
        ts (torch.tensor): time mesh
        solver (AbstractOdeSolver): ODE solver used for integration

    Returns:
        torch.tensor : trajectory from the integrated DDE
    """
    params = find_parameters(func)
    ys = nddeint_ACA.apply(history, func, ts, solver, *params)
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
