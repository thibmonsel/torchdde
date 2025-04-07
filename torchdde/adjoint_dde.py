from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from torchdde.global_interpolation.linear_interpolation import TorchLinearInterpolator
from torchdde.integrate import _integrate_dde, _integrate_ode
from torchdde.misc import TupleTensorTransformer
from torchdde.solver.base import AbstractOdeSolver
from torchdde.step_size_controller.base import AbstractStepSizeController
from torchdde.step_size_controller.constant import ConstantStepSizeController


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
        *params,  # type: ignore
    ) -> Float[torch.Tensor, "batch time ..."]:
        # Saving parameters for backward()
        ctx.history_func = history_func
        ctx.stepsize_controller = stepsize_controller
        ctx.solver = solver
        ctx.func = func
        ctx.func_args = func_args
        ctx.ts = ts
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.max_steps = max_steps

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
        state_interpolator = ctx.ys_interpolator

        # computing y'(t-tau) for the contribution of
        # delay parameters in the loss w.r.t to the parameters
        grad_ys = torch.gradient(ctx.ys, dim=1, spacing=(ctx.ts,))[0]
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
        grad_ys_history_func = torch.gradient(
            ys_history_eval, spacing=(ts_history,), dim=1
        )[0]
        grad_ys = torch.cat((grad_ys_history_func, grad_ys), dim=1)

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

        def augment_dyn(t, aug_state, func_args):
            adjoint_y, *params_inc = transformer.unflatten(aug_state)
            y_t = torch.autograd.Variable(
                state_interpolator(t) if t > ctx.t0 else ctx.history_func(t),
                requires_grad=True,
            )
            h_t_minus_tau = [
                (
                    torch.autograd.Variable(
                        state_interpolator(t - tau), requires_grad=True
                    )
                    if t - tau > ctx.t0
                    else torch.autograd.Variable(
                        ctx.history_func(t - tau), requires_grad=True
                    )
                )
                for tau in ctx.func.delays
            ]
            func_t = ctx.func(t, y_t, func_args, history=h_t_minus_tau)
            # This correspond to the both terms :
            # \lambda_t \partial{f_\theta(t)}{g} in adjoint dynamics
            # \lambda_t \partial{f_\theta(t)}{\theta} and in loss equation
            rhs_adjoint_1, *params_inc = torch.autograd.grad(
                func_t,
                (y_t,) + params,
                -adjoint_y,
                retain_graph=True,
                allow_unused=True,
            )

            # we need to add the second term of rhs in rhs_adjoint computation
            delay_derivative_inc = torch.zeros_like(ctx.func.delays)[..., None]
            for idx, tau_i in enumerate(ctx.func.delays):
                # This is computing part of the contribution of the gradient's
                # loss w.r.t the parameters
                # \lambda(t) \partial{f_\theta(t)}{g}
                # where g(t) = y(t-tau_i)
                params_inc2 = torch.autograd.grad(
                    func_t,
                    h_t_minus_tau[idx],
                    -adjoint_y,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                if params_inc2 is None:
                    pass
                else:
                    delay_derivative_inc[idx] += torch.sum(
                        params_inc2 * grad_ys[:, -1 - i],
                        dim=(tuple(range(len(params_inc2.shape)))),
                    )

                # if t+ tau_i > T then \lambda(t+tau_i) = 0
                # computing second term of the adjoint dynamics
                # \lambda_t \partial{f_\theta(t)}{g}
                if t < ctx.t1 - tau_i:
                    adjoint_t_plus_tau = adjoint_interpolator(t + tau_i)
                    y_t_plus_tau = state_interpolator(t + tau_i)
                    history = [
                        (
                            state_interpolator(t + tau_i - tau_j)
                            if t + tau_i - tau_j > ctx.ts[0]
                            else ctx.history_func(t + tau_i - tau_j)
                        )
                        for tau_j in ctx.func.delays
                    ]
                    history[idx] = y_t
                    func_t_plus_tau_i = ctx.func(
                        t + tau_i, y_t_plus_tau, func_args, history=history
                    )

                    # This correspond to the term
                    # \lambda(t+tau_i) \partial{f_\theta(t+tau_i)}{y_i}
                    # where y_i(t) = g(t-\tau_i)
                    rhs_adjoint_2 = torch.autograd.grad(
                        func_t_plus_tau_i, y_t, -adjoint_t_plus_tau
                    )[0]
                    rhs_adjoint_1 += rhs_adjoint_2

            params_inc = tuple(
                [
                    -param
                    if param is not None
                    else torch.zeros_like(transformer.original_shapes[i])
                    for i, param in enumerate(params)
                ]
            )

            return transformer.flatten(
                (
                    rhs_adjoint_1,
                    -delay_derivative_inc.squeeze(1) + params_inc[0],
                    *params_inc[1:],
                )
            )

        # computing the adjoint dynamics
        current_num_steps = 0
        for i in range(len(ts) - 1, 0, -1):
            current_num_steps += 1
            if current_num_steps > ctx.max_steps:
                raise RuntimeError("Maximum number of steps reached")

            t0, t1 = ts[i], ts[i - 1]
            dt = t1 - t0
            dt = torch.clamp(dt, max=torch.min(ctx.func.delays))
            with torch.enable_grad():
                aug_state[0] += grad_output[:, i]
                adjoint_interpolator.add_point(t0, aug_state[0])
                aug_state = transformer.flatten(aug_state)
                new_aug_state, _ = _integrate_ode(
                    augment_dyn,
                    t0,
                    t1,
                    t1[None],
                    aug_state,
                    func_args,
                    solver,
                    stepsize_controller,
                    dt,
                    ctx.max_steps,
                )
                aug_state = transformer.unflatten(new_aug_state)

        params_incr = aug_state[1:]
        tuple_nones = (None, None, None, None, None, None, None, None, None, None)
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
