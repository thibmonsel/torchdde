from typing import Any, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.integrate import _integrate_ode
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller.base import AbstractStepSizeController
from torchdde.step_size_controller.constant import ConstantStepSizeController


class odeint_ACA(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        func: torch.nn.Module,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        ts: Float[torch.Tensor, " time"],
        y0: Float[torch.Tensor, "batch ..."],
        args: Any,
        solver: AbstractOdeSolver,
        stepsize_controller: AbstractStepSizeController,
        dt0: Optional[Float[torch.Tensor, ""]] = None,
        max_steps: Optional[int] = 2048,
        *params,  # type: ignore
    ) -> Float[torch.Tensor, "batch time ..."]:
        # Saving parameters for backward()
        ctx.func = func
        ctx.ts = ts
        ctx.y0 = y0
        ctx.solver = solver
        ctx.stepsize_controller = stepsize_controller
        ctx.max_steps = max_steps

        with torch.no_grad():
            ctx.save_for_backward(*params)
            ys, _ = _integrate_ode(
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
            )
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
        for i in range(len(ts) - 1, 0, -1):
            t0, t1 = ts[i], ts[i - 1]
            dt = t1 - t0
            y_t = torch.autograd.Variable(ys[:, i], requires_grad=True)
            with torch.enable_grad():
                out = ctx.func(ts[i], y_t, args)
                adj_dyn = lambda t, adj_y, args: torch.autograd.grad(
                    out, y_t, -adj_y, retain_graph=True
                )[0]
                adjoint_state, _ = _integrate_ode(
                    adj_dyn,
                    t0,
                    t1,
                    t1[None],
                    adjoint_state,
                    args,
                    solver,
                    stepsize_controller,
                    dt,
                    ctx.max_steps,
                )
                adjoint_state = adjoint_state.squeeze(dim=1)
                adjoint_state = adjoint_state - grad_output[:, i]
                param_inc = torch.autograd.grad(
                    out, params, -adjoint_state, retain_graph=True
                )

            if out2 is None:
                out2 = tuple([dt.abs() * p for p in param_inc])
            else:
                for _1, _2 in zip([*out2], [*param_inc]):
                    _1 += dt.abs() * _2
        return (  # type: ignore
            None,
            None,
            None,
            None,
            adjoint_state,
            None,
            None,
            None,
            None,
            None,
            *out2,  # type: ignore
        )


def odesolve_adjoint(
    func: torch.nn.Module,
    t0: Float[torch.Tensor, ""],
    t1: Float[torch.Tensor, ""],
    ts: Float[torch.Tensor, " time"],
    y0: Float[torch.Tensor, "batch ..."],
    args: Any,
    solver: AbstractOdeSolver,
    stepsize_controller: AbstractStepSizeController = ConstantStepSizeController(),
    dt0: Optional[Float[torch.Tensor, ""]] = None,
    max_steps: Optional[int] = 2048,
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
    zs = odeint_ACA.apply(
        func, t0, t1, ts, y0, args, solver, stepsize_controller, dt0, max_steps, *params
    )
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
