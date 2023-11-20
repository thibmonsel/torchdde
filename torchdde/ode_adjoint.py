import functools

import numpy as np
import scipy
import scipy.integrate as sciinteg
import torch
import torch.nn as nn

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


# Used for float comparisons


class odeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, func, ts, solver, *params):
        # Saving parameters for backward()
        ctx.func = func
        ctx.ts = ts
        ctx.y0 = y0
        ctx.solver = solver
        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys = solver.integrate(func, ts, y0)
        ctx.ys = ys
        return ys

    @staticmethod
    def backward(ctx, *grad_y):
        # This function implements the adjoint gradient estimation method for NODEs
        # grad_output holds the gradient of the loss w.r.t. each evaluation step
        grad_output = grad_y[0]

        # h is the value of the forward time step
        h = -(ctx.ts[1] - ctx.ts[0])
        # Retrieving the time mesh and the corresponding states created in forward()
        ys = ctx.ys
        ts = ctx.ts

        params = ctx.saved_tensors
        solver = ctx.solver
        # The last step of the time mesh is an evaluation step, thus the adjoint state which corresponds to the
        # gradient of the loss w.r.t. the evaluation states is initialised with the gradient corresponding to
        # the last evaluation time.

        adjoint_state = torch.zeros_like(grad_output[:, -1])
        # The adjoint state as well as the parameters' gradient are integrated backwards in time.
        # Following the Adaptive Checkpoint Adjoint method, the time steps and corresponding states of the forward
        # integration are re-used by going backwards in the time mesh.

        out2 = None
        for i, t in enumerate(reversed(ts)):
            # adjoint_state -= grad_output[:, -j - 1]
            # Backward Integrating the adjoint state and the parameters' gradient between time i and i-1
            y_t = torch.autograd.Variable(ys[:, i - 1], requires_grad=True)

            with torch.enable_grad():
                out = ctx.func(t, y_t)
                adj_fn = lambda t, adj_y: torch.autograd.grad(
                    out, y_t, -adj_y, retain_graph=True
                )[0]
                adjoint_state = solver.step(adj_fn, t, adjoint_state, h)

                # Computing the increment to the parameters' gradient corresponding to the current time step
                param_inc = torch.autograd.grad(
                    out, params, -adjoint_state, retain_graph=True
                )

                # The following line corresponds to an integration step of the adjoint state

            # incrementing the parameters' grad
            if out2 is None:
                out2 = tuple([-h * p for p in param_inc])
            else:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 -= h * _2

        return adjoint_state, None, None, None, *out2


def odesolve_adjoint(z0, func, ts, solver):
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a

    # The parameters for which a gradient should be computed are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    # flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    zs = odeint_ACA.apply(z0, func, ts, solver, *params)
    return zs


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
