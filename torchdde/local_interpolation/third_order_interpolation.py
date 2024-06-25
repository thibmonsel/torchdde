from typing import Dict, Optional, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.fourth_order_interpolation import linear_rescale


class ThirdOrderPolynomialInterpolation:
    """Hermite Polynomial third order interpolation on [t0, t1].

    `coefficients` holds the coefficients of a fourth-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "..."]],
    ):
        self.t0 = t0
        self.t1 = t1
        self.dt = t1 - t0
        self.coeffs = self._calculate(dense_info)

    def _calculate(self, dense_info: Dict[str, Float[torch.Tensor, "..."]]):
        _k0, _k1 = dense_info["k"][0], dense_info["k"][-1]
        _a = _k0 + _k1 + 2 * dense_info["y0"] - 2 * dense_info["y1"]
        _b = -2 * _k0 - _k1 - 3 * dense_info["y0"] + 3 * dense_info["y1"]
        return torch.stack([_a, _b, _k0, dense_info["y0"]], dim=1)

    def __call__(
        self,
        t: Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]],
        t1: Optional[Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]]] = None,
        left: Optional[bool] = True,
    ) -> Float[torch.Tensor, "batch ..."]:
        del left
        if t1 is not None:
            return self.__call__(t1) - self.__call__(t)

        t = linear_rescale(self.t0, t, self.t1)
        t_polynomial = torch.pow(
            t * torch.ones((4,), device=t.device, dtype=t.dtype),
            exponent=torch.flip(
                torch.arange(4, device=t.device), dims=(0,)
            ),  # pyright : ignore
        )
        return torch.einsum("c, bcf -> bf", t_polynomial, self.coeffs)

    def __repr__(self):
        return (
            f"ThirdOrderHermiteInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coeffs})"
        )
