from typing import Any, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.integrate import _integrate_ode
from torchdde.misc import TupleTensorTransformer
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
        ctx.dt0 = dt0
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
        ys = ctx.ys
        ts = ctx.ts
        dt0 = ctx.dt0
        args = ctx.args

        solver = ctx.solver
        stepsize_controller = ctx.stepsize_controller
        params = ctx.saved_tensors
        # aug_state will hold the [y_t, adjoint_state, params_incr]
        aug_state = [torch.zeros_like(ys[:, -1]), torch.zeros_like(ys[:, -1])]
        aug_state.extend([torch.zeros_like(param) for param in params])
        transformer = TupleTensorTransformer.from_tuple(aug_state)

        def augmented_dyn(t, aug_state, args):
            y_t, adjoint_state, *params_inc = transformer.unflatten(aug_state)
            out = ctx.func(t, y_t, args)
            adjoint_state, *params_inc = torch.autograd.grad(
                out,
                (y_t,) + params,
                -adjoint_state,
                retain_graph=True,
                allow_unused=True,
            )
            return transformer.flatten((y_t, adjoint_state, *params_inc))

        for i in range(len(ts) - 1, 0, -1):
            t0, t1 = ts[i], ts[i - 1]
            dt0 = t1 - t0
            y_t = torch.autograd.Variable(ys[:, i], requires_grad=True)

            aug_state[0] = y_t
            aug_state[1] += grad_output[:, i]

            with torch.enable_grad():
                aug_state[0] = y_t
                aug_state = transformer.flatten(aug_state)
                new_aug_state, _ = _integrate_ode(
                    augmented_dyn,
                    t0,
                    t1,
                    t1[None],
                    aug_state,
                    args,
                    solver,
                    stepsize_controller,
                    dt0,
                    ctx.max_steps,
                )
                aug_state = transformer.unflatten(new_aug_state)

        adjoint_state = aug_state[1]
        params_incr = aug_state[2:]
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
            *params_incr,  # type: ignore
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
