from abc import ABC, abstractmethod

import torch
from jaxtyping import Float


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the step method.
    """

    @abstractmethod
    def step(
        self,
        func: torch.nn.Module,
        t: Float[torch.Tensor, "1"],
        y: Float[torch.Tensor, "batch ..."],
        dt: Float[torch.Tensor, "1"],
        args=None,
        has_aux=False,
    ) -> torch.Tensor:
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
        func: torch.nn.Module,
        ts: Float[torch.Tensor, "time"],
        y0: Float[torch.Tensor, "batch ..."],
        args=None,
        has_aux=False,
    ) -> torch.Tensor:
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
        current_y = y0
        for current_t in ts[1:]:
            y = self.step(func, current_t, current_y, dt, args=args, has_aux=has_aux)
            current_y = y
            ys = torch.cat((ys, torch.unsqueeze(current_y, dim=1)), dim=1)
        return ys


class Euler(AbstractOdeSolver):
    """Euler's method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, args, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, args, has_aux)
            return y + dt * k1, aux
        else:
            return y + dt * func(t, y, args)


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
    def _residual(f, t, y, dt, y_sol, args):
        f_sol = f(t, y_sol, args)
        return torch.sum((y_sol - y - dt * f_sol) ** 2)

    def step(self, func, t, y, dt, args, has_aux=False):
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

        def closure():
            opt.zero_grad()
            residual = ImplicitEuler._residual(func, t, y, dt, y_sol, args)
            (y_sol.grad,) = torch.autograd.grad(
                residual, y_sol, only_inputs=True, allow_unused=False
            )
            return residual

        opt.step(closure)
        if has_aux:
            _, aux = func(t, y, args, has_aux)
            return y_sol, aux
        else:
            return y_sol


class RK2(AbstractOdeSolver):
    """2nd order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, args, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, args, has_aux)
            k2 = func(t + dt, y + dt * k1, args)
            return y + dt / 2 * (k1 + k2), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + dt, y + dt * k1, args)
            return y + dt / 2 * (k1 + k2)


class Ralston(AbstractOdeSolver):
    """Ralston's method (2nd order)"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, args, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, args, has_aux)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1, args)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2)


class RK4(AbstractOdeSolver):
    """4th order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, args, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, args, has_aux)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4 = func(t + dt, y + dt * k3, args)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4), aux
        else:
            k1 = func(t, y, args)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1, args)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2, args)
            k4 = func(t + dt, y + dt * k3, args)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)


# solver = ImplicitEuler()
# ts = torch.linspace(0, 20, 201)
# ys = solver.integrate(lambda t, y, args: -y, ts, torch.tensor([1.0]), None)
# import matplotlib.pyplot as plt


# plt.plot(ts, ys[0].detach().numpy())
# plt.show()
