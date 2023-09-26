import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.solver.dde_solver import DDESolver
from torchdde.solver.ode_solver import *

#
# Implements a version of the NeuralODE adjoint optimisation algorithm, with the Adaptive Checkpoint Adjoint method
#
# Original NODE : https://arxiv.org/abs/1806.07366
# ACA version : https://arxiv.org/abs/2006.02493
#
# The forward integration is done through the scipy IVP solver.
# The backpropagation is based on the timesteps chosen by the scipy solver and only requires a step function for
# the relevant scheme to be added in the code.
# Any explicit scheme available with the scipy solver can be easily added here.

#
# NB : This code is heavily based on the torch_ACA package (https://github.com/juntang-zhuang/torch_ACA)


class nddeint2_ACA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, history_func, func, ts, *params):
        # Saving parameters for backward()
        ctx.history_func = history_func
        ctx.func = func
        ctx.ts = ts
        with torch.no_grad():
            ctx.save_for_backward(*params)

            # Simulation
            solver = RK4()
            dde_solver = DDESolver(solver, func.delays)
            ys, ys_interpolator = dde_solver.integrate(func, ts, history_func)

        ctx.ys_interpolation = ys_interpolator
        ctx.ys = ys
        return ctx.ys

    @staticmethod
    def backward(ctx, *grad_y):
        # https://www.researchgate.net/publication/255686149_Adjoint_Sensitivity_Analysis_of_Neutral_Delay_Differential_Models
        # This function implements the adjoint gradient estimation method for NODEs

        # grad_output holds the gradient of the loss w.r.t. to the output of the forward ie ys
        grad_output = grad_y[0]
        # Retrieving the time mesh and the corresponding states created in forward()
        # allstates = ctx.allstates
        ts = ctx.ts
        dt = -(ts[1] - ts[0])
        # f_params holds the NDDE parameters for which a gradient will be computed
        params = ctx.saved_tensors
        state_interpolator = ctx.ys_interpolation
        history_func = ctx.history_func
        # This is the adjoint state at t = T
        T = ts[-1]
        adjoint_state = grad_output[:, -1]

        adjoint_history_func = (
            lambda t: adjoint_state
        )  # if t == T else torch.zeros_like(adjoint_state)
        adjoint_interpolator = TorchLinearInterpolator(
            torch.tensor([T]), torch.unsqueeze(adjoint_state, dim=1)
        )
        tau = max(ctx.func.delays)

        def adjoint_dyn(t, adjoint_y):
            h_t = torch.autograd.Variable(state_interpolator(t), requires_grad=True)
            h_t_minus_tau = (
                state_interpolator(t - tau) if t - tau >= ctx.ts[0] else history_func(t)
            )
            out = ctx.func(t, h_t, history=[h_t_minus_tau])
            rhs_adjoint_inc = torch.autograd.grad(
                out, h_t, -adjoint_y, retain_graph=True
            )[0]
            rhs_adjoint = rhs_adjoint_inc

            if t < T - tau:
                adjoint_t_plus_tau = adjoint_interpolator(t + tau)
                h_t_plus_tau = state_interpolator(t + tau)
                out_other = ctx.func(t, h_t_plus_tau, history=[h_t])

                rhs_adjoint_inc_k1 = torch.autograd.grad(
                    out_other, h_t, -adjoint_t_plus_tau, retain_graph=True
                )[0]

                rhs_adjoint += rhs_adjoint_inc_k1

            return rhs_adjoint

        solver = RK4()
        current_adjoint = adjoint_history_func(ts[-1])

        with torch.enable_grad():
            # computing the adjoint dynamics
            for j, current_t in enumerate(reversed(ts)[:-1]):
                adj = solver.step(adjoint_dyn, current_t, current_adjoint, dt)
                current_adjoint = adj + grad_output[:, -j - 1]
                adjoint_interpolator.add_point(current_t + dt, adj)

            # Computing the gradient of the loss w.r.t. the parameters
            def loss_dynamics(t):
                h_t = torch.autograd.Variable(state_interpolator(t), requires_grad=True)
                h_t_minus_tau = (
                    state_interpolator(t - tau)
                    if t - tau >= ctx.ts[0]
                    else history_func(t)
                )
                out = ctx.func(t, h_t, history=[h_t_minus_tau])
                params_dyn = torch.autograd.grad(
                    out, params, adjoint_interpolator(t), retain_graph=True
                )
                return params_dyn

            out2 = tuple([dt * loss_dynamics(t) for t in ts])
            # out2 = None
            # dt_params = ts[1] - ts[0]
            # aux = []
            # for j, t in enumerate(ts[:-1]) :
            #     h_t = torch.autograd.Variable(state_interpolator(t), requires_grad=True)
            #     h_t_minus_tau = state_interpolator(t - tau) if t - tau >= ctx.ts[0] else history_func(t)
            #     out = ctx.func(t, h_t, history=[h_t_minus_tau])
            #     param_derivative_inc = torch.autograd.grad(out, params, adjoint_interpolator(t), retain_graph=True)
            #     aux.append(*param_derivative_inc)
            #     if out2 is None:
            #         out2 = tuple([dt_params * p for p in param_derivative_inc])
            #     else:
            #         for _1, _2 in zip([*out2], [*param_derivative_inc]):
            #             _1 = _1 + dt_params * _2
        # print(torch.stack(aux).shape)
        # plt.plot(torch.stack(aux)[:, 0])
        # plt.title("dyn of los")
        # plt.show()
        # print(adjoint_interpolator.ys[0].shape)
        # plt.plot(ts, adjoint_interpolator.ys[0])
        # plt.title("Adjoint")
        # plt.show()
        # print("Real params -0.5 and 1 ")
        # print("model params", *params)
        # print("DL_dtheta", *out2)

        return None, None, None, *out2


class nddeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, history_func, func, ts, *params):
        # Saving parameters for backward()
        ctx.history_func = history_func
        ctx.func = func
        ctx.ts = ts
        with torch.no_grad():
            ctx.save_for_backward(*params)

            # Simulation
            solver = Euler()
            dde_solver = DDESolver(solver, func.delays)
            ys, ys_interpolator = dde_solver.integrate(func, ts, history_func)

        ctx.ys_history_func = history_func
        ctx.ys_interpolator = ys_interpolator
        ctx.ys = ys
        return ctx.ys

    @staticmethod
    def backward(ctx, *grad_y):
        # https://www.researchgate.net/publication/255686149_Adjoint_Sensitivity_Analysis_of_Neutral_Delay_Differential_Models
        # This function implements the adjoint gradient estimation method for NODEs
        # grad_output holds the gradient of the loss w.r.t. each evaluation step
        grad_output = grad_y[0]

        T = ctx.ts[-1]
        dt = ctx.ts[1] - ctx.ts[0]

        params = ctx.saved_tensors
        state_interpolator = ctx.ys_interpolator

        adjoint_state = torch.zeros_like(grad_output[:, -1])
        adjoint_ys_final = -grad_output[:, -1].reshape(
            adjoint_state.shape[0], 1, *adjoint_state.shape[1:]
        )

        # adjoint history shape [N, D]
        # create an adjoint interpolator that is defined from [T, T+ dt] for initialization purposes
        # adjoint at T=t is equal to the gradient of the loss w.r.t. the evaluation state at t=T and the t> T is 0
        adjoint_interpolator = TorchLinearInterpolator(
            torch.unsqueeze(ctx.ts[-1], dim=0),
            adjoint_ys_final,
            # device=adjoint_ys_final.device,
        )

        # The adjoint state as well as the parameters' gradient are integrated backwards in time.
        # Following the Adaptive Checkpoint Adjoint method, the time steps and corresponding states of the forward
        # integration are re-used by going backwards in the time mesh.
        stacked_params = None
        for j, t in enumerate(reversed(ctx.ts)):
            # Backward Integrating the adjoint state and the parameters' gradient between time i and i-1
            adjoint_state -= grad_output[:, -j - 1]
            adjoint_interpolator.add_point(t, adjoint_state)
            with torch.enable_grad():
                # Taking a step with the NODE function to build a graph which will be differentiated
                # so as to integrate the adjoint state and the parameters' gradient
                rhs_adjoint = 0.0

                # correspond to h_t
                h_t = torch.autograd.Variable(state_interpolator(t), requires_grad=True)

                # we are in the case where t > T - tau
                h_t_minus_tau = [
                    state_interpolator(t - tau) if t - tau >= ctx.ts[0] else ctx.ys_history_func(t-tau)
                    for tau in ctx.func.delays
                ]
                out = ctx.func(t, h_t, history=h_t_minus_tau)
                rhs_adjoint_inc_k1 = torch.autograd.grad(
                    out, h_t, -adjoint_state, retain_graph=True
                )[0]

                rhs_adjoint += rhs_adjoint_inc_k1

                # we need to add the the second term of rhs too in rhs_adjoint computation
                for idx, tau_i in enumerate(ctx.func.delays):
                    if t < T - tau_i:
                        adjoint_t_plus_tau = adjoint_interpolator(t + tau_i)
                        h_t_plus_tau = state_interpolator(t + tau_i)
                        history = [
                            state_interpolator(t + tau_i - tau_j)
                            if t + tau_i - tau_j >= ctx.ts[0]
                            else ctx.ys_history_func(t+ tau_i- tau_j)
                            for tau_j in ctx.func.delays
                        ]
                        history[idx] = h_t
                        out_other = ctx.func(
                            t + tau_i, h_t_plus_tau, history=history
                        )  
                        rhs_adjoint_inc_k1 = torch.autograd.grad(
                            out_other, h_t, -adjoint_t_plus_tau
                        )[0]

                        rhs_adjoint = rhs_adjoint + rhs_adjoint_inc_k1

                param_derivative_inc = torch.autograd.grad(
                    out, params, -adjoint_state, retain_graph=False
                )

                if stacked_params is None:
                    stacked_params = tuple(
                        [torch.unsqueeze(p, dim=-1) for p in param_derivative_inc]
                    )
                else:
                    stacked_params = tuple(
                        [
                            torch.concat([_1, torch.unsqueeze(_2, dim=-1)], dim=-1)
                            for _1, _2 in zip(stacked_params, param_derivative_inc)
                        ]
                    )

                adjoint_state = adjoint_state - dt * rhs_adjoint

        cum_sum = tuple([dt * torch.cumsum(p, dim=-1) for p in stacked_params])
        sum_cum_sum = tuple([torch.trapezoid(p, dim=-1) for p in cum_sum])

        return None, None, None, *sum_cum_sum
        # return None, None, None, *out2


def nddesolve_adjoint(history, func, options):
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a

    # The parameters for which a gradient should be computed are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    # flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    # zs = nddeint2_ACA.apply(history, func, options, *params)
    zs = nddeint_ACA.apply(history, func, options, *params)
    return zs


def flatten_grad_params(params):
    # Parameters for which a grad is required are flattened and returned as a list
    flat_params = []
    for p in params:
        if p.requires_grad:
            flat_params.append(p.contiguous().view(-1))

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def flatten_params(params):
    # values in the params tuple are flattened and returned as a list
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def get_integration_options(n0, n1, dt, substeps):
    sub_dt = float(dt / substeps)
    nSteps = substeps * (n1 - n0)

    integration_options = {
        "t0": n0 * dt,
        "dt": sub_dt,
        "nSteps": nSteps,
        "eval_idx": np.arange(0, nSteps + 1, substeps),
    }

    return integration_options


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