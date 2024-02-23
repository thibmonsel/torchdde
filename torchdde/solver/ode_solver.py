from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import torch
from jaxtyping import Float


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the step method.
    """

    @abstractmethod
    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        r"""ODE's stepping definition

        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `t`: Current time step t
        - `y`: Current state y
        - `dt`: Stepsize dt
        - `has_aux`: Whether the model has an auxiliary output.

        **Returns:**

        Integration result at time `t+dt`
        """
        pass

    def integrate(
        self,
        func: Union[torch.nn.Module, Callable],
        ts: Float[torch.Tensor, " time"],
        y0: Float[torch.Tensor, "batch ..."],
        args: Any,
    ) -> Float[torch.Tensor, "batch ..."]:
        r"""Integrate a system of ODEs.
        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `ts`: Integration span
        - `y0`: Initial condition
        - `has_aux`: Whether the model has an auxiliary output.

        **Returns:**

        Integration result over `ts`
        """

        dt = ts[1] - ts[0]
        ys = torch.unsqueeze(y0.clone(), dim=1)
        for current_t in ts[1:]:
            y, _ = self.step(func, current_t, ys[:, -1], dt, args, has_aux=False)
            ys = torch.cat((ys, torch.unsqueeze(y, dim=1)), dim=1)
        return ys


class Euler(AbstractOdeSolver):
    """Euler's method"""

    def __init__(self):
        super().__init__()

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        if has_aux:
            k1, aux = func(t, y, args)
            return y + dt * k1, aux
        else:
            return y + dt * func(t, y, args), None


class ImplicitEuler(AbstractOdeSolver):
    """ImplicitEuler Euler's method"""

    #  Credits to the TorchDyn team for the implementation of the implicit Euler method.
    # Savagely copied from:
    # https://github.com/DiffEqML/torchdyn/blob/95cc74b0e35330b03d2cd4d875df362a93e1b5ea/torchdyn/numerics/solvers/ode.py#L181

    def __init__(self):
        super().__init__()
        self.opt = torch.optim.LBFGS
        self.max_iters = 100

    @staticmethod
    def _residual(
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        y_sol: Float[torch.Tensor, "batch ..."],
        args: Any,
        has_aux=False,
    ) -> Float[torch.Tensor, ""]:
        if has_aux:
            f_sol, _ = func(t, y_sol, args)
        else:
            f_sol = func(t, y_sol, args)
        return torch.sum((y_sol - y - dt * f_sol) ** 2)

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        y_sol = y.clone()
        y_sol = torch.nn.Parameter(data=y_sol)
        opt = self.opt(
            [y_sol],
            lr=1,
            max_iter=self.max_iters,
            max_eval=10 * self.max_iters,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Float[torch.Tensor, ""]:
            opt.zero_grad()
            residual = ImplicitEuler._residual(
                func, t, y, dt, y_sol, args, has_aux=has_aux
            )
            (y_sol.grad,) = torch.autograd.grad(
                residual, y_sol, only_inputs=True, allow_unused=False
            )
            return residual

        opt.step(closure)  # type: ignore
        if has_aux:
            _, aux = func(t, y, args)
            return y_sol, aux
        else:
            return y_sol, None


class RK2(AbstractOdeSolver):
    """2nd order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        if has_aux:
            k1, aux = func(t, y, args)
            k2, _ = func(t + dt, y + dt * k1, args)
            return y + dt / 2 * (k1 + k2), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + dt, y + dt * k1, args)
            return y + dt / 2 * (k1 + k2), None


class Ralston(AbstractOdeSolver):
    """Ralston's method (2nd order)"""

    def __init__(self):
        super().__init__()

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        if has_aux:
            k1, aux = func(t, y, args)
            k2, _ = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), None


class RK4(AbstractOdeSolver):
    """4th order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Union[Float[torch.Tensor, ""], float],
        args: Any,
        has_aux=False,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Any]:
        if has_aux:
            k1, aux = func(t, y, args)
            k2, _ = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3, _ = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4, _ = func(t + dt, y + dt * k3, args)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4 = func(t + dt, y + dt * k3, args)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4), None
