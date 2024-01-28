import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.solver.ode_solver import *


class odeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y0, func, ts, args, solver, *params):
        # Saving parameters for backward()
        ctx.func = func
        ctx.ts = ts
        ctx.y0 = y0
        ctx.solver = solver
        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys = solver.integrate(func, ts, y0, args)
        ctx.ys = ys
        ctx.args = args
        return ys

    @staticmethod
    def backward(ctx, *grad_y):
        # grad_output holds the gradient of the loss w.r.t. each evaluation step
        grad_output = grad_y[0]

        dt = ctx.ts[1] - ctx.ts[0]
        ys = ctx.ys
        ts = ctx.ts
        args = ctx.args

        solver = ctx.solver
        params = ctx.saved_tensors
        adjoint_state = grad_output[:, -1]

        out2 = None
        for i, current_t in enumerate(reversed(ts)):
            y_t = torch.autograd.Variable(ys[:, -i - 1], requires_grad=True)

            with torch.enable_grad():
                out = ctx.func(current_t, y_t, args)
                adj_dyn = lambda t, adj_y: torch.autograd.grad(
                    out, y_t, -adj_y, retain_graph=True
                )[0]
                adjoint_state = solver.step(adj_dyn, current_t, adjoint_state, -dt)
                adjoint_state -= grad_output[:, -i - 1]

                param_inc = torch.autograd.grad(
                    out, params, -adjoint_state, retain_graph=True
                )
            if out2 is None:
                out2 = tuple([dt * p for p in param_inc])
            else:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 += dt * _2

        out = adjoint_state, None, None, None, *out2

        return out


def odesolve_adjoint(
    z0: Float[torch.Tensor, "batch ..."],
    func: torch.nn.Module,
    ts: Float[torch.Tensor, "time ..."],
    args,
    solver: AbstractOdeSolver,
) -> Float[torch.Tensor, "batch time ..."]:
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a

    # The parameters for which a gradient should be computed are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    # flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    zs = odeint_ACA.apply(z0, func, ts, args, solver, *params)
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
