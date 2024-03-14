from typing import Any, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.integrate import _integrate
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller.base import AbstractStepSizeController
from torchdde.step_size_controller.constant import ConstantStepSizeController


class odeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        y0: Float[torch.Tensor, "batch ..."],
        func: torch.nn.Module,
        ts: Float[torch.Tensor, " time"],
        args: Any,
        solver: AbstractOdeSolver,
        stepsize_controller: AbstractStepSizeController,
        *params,  # type: ignore
    ) -> Float[torch.Tensor, "batch time ..."]:
        # Saving parameters for backward()
        ctx.func = func
        ctx.ts = ts
        ctx.y0 = y0
        ctx.solver = solver
        ctx.stepsize_controller = stepsize_controller

        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys, _ = _integrate(func, solver, ts, y0, args, stepsize_controller)
        ctx.ys = ys
        ctx.args = args
        return ys

    @staticmethod
    def backward(ctx, *grad_y):  # type: ignore
        # grad_output holds the gradient of the
        # loss w.r.t. each evaluation step
        grad_output = grad_y[0]
        dt = ctx.ts[1] - ctx.ts[0]
        ys = ctx.ys
        ts = ctx.ts
        args = ctx.args

        solver = ctx.solver
        stepsize_controller = ctx.stepsize_controller
        params = ctx.saved_tensors
        adjoint_state = grad_output[:, -1]

        out2 = None
        tnext, controller_state = stepsize_controller.init(
            ctx.func, ts[-1], ts[-2], adjoint_state, -dt, args, solver.order()
        )
        tprev = ts[-1]
        for i, current_t in enumerate(reversed(ts)):
            y_t = torch.autograd.Variable(ys[:, -i - 1], requires_grad=True)

            with torch.enable_grad():
                out = ctx.func(current_t, y_t, args)
                adj_dyn = lambda t, adj_y, args: torch.autograd.grad(
                    out, y_t, -adj_y, retain_graph=True
                )[0]
                adjoint_candidate, adjoint_error, _, _ = solver.step(
                    adj_dyn, tprev, adjoint_state, controller_state, args
                )
                (
                    keep_step,
                    tprev,
                    tnext,
                    controller_state,
                ) = stepsize_controller.adapt_step_size(
                    adj_dyn,
                    tprev,
                    tnext,
                    adjoint_state,
                    adjoint_candidate,
                    args,
                    solver.order(),
                    adjoint_error,
                    controller_state,
                )

                adjoint_state = adjoint_candidate if keep_step else ys[:, -1]
                adjoint_state = adjoint_state - grad_output[:, -i - 1]

                param_inc = torch.autograd.grad(
                    out, params, -adjoint_state, retain_graph=True
                )
            # Adding last term in order to get a trapz rule
            # estimate of the grad wtr to the parameters
            # trapezoid is h/2 * (f(a) + f(b)) + [f(x1) + ... + f(xn-1)]
            # compared to rectangle rule is h * (f(a) + f(b) + f(x1) + ... + f(xn-1))
            # This could be improved by using intermediate stages by RK solvers
            if out2 is None:
                out2 = tuple([dt / 2 * p for p in param_inc])
            elif current_t == ts[0]:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 += dt / 2 * _2
            else:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 += dt * _2

        return adjoint_state, None, None, None, None, None, *out2  # type: ignore


def odesolve_adjoint(
    y0: Float[torch.Tensor, "batch ..."],
    func: torch.nn.Module,
    ts: Float[torch.Tensor, "time ..."],
    args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController = ConstantStepSizeController(),
) -> Union[Float[torch.Tensor, "batch time ..."], Any]:
    # Main function to be called to integrate the NODE

    # z0 : (tensor) Initial state of the NODE
    # func : (torch Module) Derivative of the NODE
    # options : (dict) Dictionary of solver options, should at least have a

    # The parameters for which a gradient should be computed
    # are passed as a flat list of tensors to the forward function
    # The gradient returned by backward() will take the same shape.
    # flat_params = flatten_grad_params(func.parameters())
    params = find_parameters(func)

    # Forward integrating the NODE and returning the state at each evaluation step
    zs = odeint_ACA.apply(y0, func, ts, args, solver, stepsize_controller, *params)
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
