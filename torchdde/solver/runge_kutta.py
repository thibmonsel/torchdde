import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.base import AbstractLocalInterpolation
from torchdde.solver.base import AbstractOdeSolver


class ButcherTableau:
    def __init__(
        self,
        c: Float[torch.Tensor, "..."],
        a: Float[torch.Tensor, "..."],
        b: Float[torch.Tensor, "..."],
        b_err: Float[torch.Tensor, "..."],
        b_other: Optional[Float[torch.Tensor, "..."]] = None,
        fsal: Optional[bool] = None,
        ssal: Optional[bool] = None,
    ):
        self.c = c
        self.a = a
        self.b = b
        self.b_err = b_err
        self.b_other = b_other

        if fsal is None:
            fsal = self.is_fsal()
        self.fsal = fsal
        if ssal is None:
            ssal = self.is_ssal()
        self.ssal = ssal

    @staticmethod
    def from_lists(
        *,
        c: List[float],
        a: List[List[float]],
        b: List[float],
        b_err: Optional[List[float]] = None,
        b_low_order: Optional[List[float]] = None,
        b_other: Optional[List[List[float]]] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if b_err is not None and not any(x != 0 for x in b_err):
            warnings.warn(
                "b_err_list is only zeros, so this isn't an embedded method like Dopri5"
            )

        assert b_err is not None or b_low_order is not None, (
            "You have to provide either the weights for the error approximation"
            " or the weights of an embedded lower-order method"
        )

        n_nodes = len(c)
        n_weights = len(b)
        assert n_nodes == n_weights
        assert len(a) == n_nodes

        # Fill a up into a full square matrix
        a_full = [row + [0.0] * (n_weights - len(row)) for row in a]

        b_coeffs = torch.tensor(b, dtype=dtype)
        if b_err is None:
            assert b_low_order is not None
            assert len(b_low_order) == n_weights
            b_low_coeffs = torch.tensor(b_low_order, dtype=dtype)
            b_err_coeffs = b_coeffs - b_low_coeffs
        else:
            b_err_coeffs = torch.tensor(b_err, dtype=dtype)

        if b_other is None:
            b_other_coeffs = None
        else:
            b_other_coeffs = torch.tensor(b_other, dtype=dtype)
            assert b_other_coeffs.ndim == 2
            assert b_other_coeffs.shape[1] == n_weights

        return ButcherTableau(
            c=torch.tensor(c, dtype=dtype),
            a=torch.tensor(a_full, dtype=dtype),
            b=b_coeffs,
            b_err=b_err_coeffs,
            b_other=b_other_coeffs,
        )

    def to(
        self, device: torch.device, time_dtype: torch.dtype, data_dtype: torch.dtype
    ) -> "ButcherTableau":
        b_other = self.b_other
        if b_other is not None:
            b_other = b_other.to(device, data_dtype)
        return ButcherTableau(
            c=self.c.to(device, time_dtype),
            a=self.a.to(device, data_dtype),
            b=self.b.to(device, data_dtype),
            b_err=self.b_err.to(device, data_dtype),
            b_other=b_other,
            fsal=self.fsal,
            ssal=self.ssal,
        )

    @property
    def n_stages(self):
        return self.c.shape[0]

    def is_fsal(self):
        """Is `f(y0)` equal to `f(y1)` from the previous step?

        If that is the case, we can reuse the result from the previous step.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all().item()
        first_node_is_t0 = (self.c[0] == 0.0).item()
        last_node_is_t1 = (self.c[-1] == 1.0).item()
        first_stage_explicit = (self.a[0, 0] == 0.0).item()
        return bool(
            is_lower_triangular
            and (self.b == self.a[-1]).all().item()
            and first_node_is_t0
            and last_node_is_t1
            and first_stage_explicit
        )

    def is_ssal(self):
        """Is the solution equal to the last stage result?

        If that is the case, we can avoid the final computation of the solution and
        return the last stage result instead.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all().item()
        last_node_is_t1 = (self.c[-1] == 1.0).item()
        last_stage_explicit = (self.a[-1, -1] == 0.0).item()
        return bool(
            is_lower_triangular
            and (self.b == self.a[-1]).all().item()
            and last_node_is_t1
            and last_stage_explicit
        )


class ExplicitRungeKutta(AbstractOdeSolver):
    def __init__(
        self,
        tableau: ButcherTableau,
        interpolation_cls: Type[AbstractLocalInterpolation],
    ):
        super().__init__()

        self.tableau = tableau
        self.interpolation_cls = interpolation_cls

    def init(
        self,
        func: Union[torch.nn.Module, Callable],
        t0: Float[torch.Tensor, ""],
        y0: Float[torch.Tensor, "batch ..."],
        dt0: Float[torch.Tensor, ""],
        func_args: Any,
        f0: Optional[Float[torch.Tensor, "batch ..."]] = None,
        *args,
        **kwargs,
    ) -> Tuple[Optional[Float[torch.Tensor, "batch ..."]]]:
        del dt0, args, kwargs
        if self.tableau.fsal:
            if f0 is None:
                prev_vf1 = func(t0, y0, func_args)
            else:
                prev_vf1 = f0
        else:
            prev_vf1 = None

        return (prev_vf1,)

    def step(
        self,
        func: Union[torch.nn.Module, Callable],
        t: Float[torch.Tensor, ""],
        y: Float[torch.Tensor, "batch ..."],
        dt: Float[torch.Tensor, ""],
        solver_state: Union[Tuple[Any, ...], None],
        func_args: Any,
        has_aux: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "batch ..."],
        Float[torch.Tensor, "batch ..."],
        dict[str, Float[torch.Tensor, "batch order"]],
        Union[Tuple[Float[torch.Tensor, "batch ..."]], None],
        Union[Float[torch.Tensor, " batch"], Any],
    ]:
        if self.tableau.fsal:
            assert solver_state is not None
            f0, *_ = solver_state
        else:
            f0 = func(t, y, func_args)
        print(f0.shape)
        y_i = y
        t_nodes = torch.addcmul(t, self.tableau.c, dt)
        k = f0.new_empty((self.tableau.n_stages, f0.shape[0], f0.shape[1]))
        k[0] = f0
        for i in range(1, self.tableau.n_stages):
            y_i = torch.einsum("j, jbf -> bf", self.tableau.a[i, :i], k[:i])
            y_i = torch.addcmul(y, dt, y_i)
            k[i] = func(t_nodes[i], y_i, func_args)

        if self.tableau.ssal:
            y1 = y_i
        else:
            y1 = y + dt * torch.einsum("s, sbf -> bf", self.tableau.b, k)

        y_error = dt * torch.einsum("s, sbf -> bf", self.tableau.b_err, k)
        dense_info = dict(y0=y, y1=y1, k=k)

        if self.tableau.fsal:
            new_solver_state = (k[-1],)
        else:
            new_solver_state = None

        return (
            y1,
            y_error,
            dense_info,
            new_solver_state,
            None,
        )

    def build_interpolation(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "batch ..."]],
    ) -> AbstractLocalInterpolation:
        return self.interpolation_cls(t0, t1, dense_info)  # type: ignore
