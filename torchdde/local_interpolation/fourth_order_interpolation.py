from typing import Dict, Optional, Union

import torch
from jaxtyping import Float


def linear_rescale(t0, t, t1):
    """
    Calculates (t - t0) / (t1 - t0), assuming t0 <= t <= t1.
    """

    cond = t0 == t1
    numerator = torch.zeros_like(t) if cond else t - t0
    denominator = torch.ones_like(t) if cond else t1 - t0
    return numerator / denominator


class FourthOrderPolynomialInterpolation:
    """Polynomial interpolation on [t0, t1].

    `coefficients` holds the coefficients of a fourth-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "..."]],
        c_mid: Float[torch.Tensor, " nb_stages"],
    ):
        self.t0 = t0
        self.t1 = t1
        self.dt = t1 - t0
        self.c_mid = c_mid
        self.coeffs = self._calculate(dense_info)

    def _calculate(
        self, dense_info: Dict[str, Float[torch.Tensor, "..."]]
    ) -> Float[torch.Tensor, "nb_stages batch ..."]:
        _ymid = dense_info["y0"] + self.dt * torch.einsum(
            "c, cbf -> bf", self.c_mid, dense_info["k"]
        )
        _f0 = self.dt * dense_info["k"][0]
        _f1 = self.dt * dense_info["k"][-1]

        _a = 2 * (_f1 - _f0) - 8 * (dense_info["y1"] + dense_info["y0"]) + 16 * _ymid
        _b = (
            5 * _f0
            - 3 * _f1
            + 18 * dense_info["y0"]
            + 14 * dense_info["y1"]
            - 32 * _ymid
        )
        _c = _f1 - 4 * _f0 - 11 * dense_info["y0"] - 5 * dense_info["y1"] + 16 * _ymid
        return torch.stack([_a, _b, _c, _f0, dense_info["y0"]], dim=1).type(
            torch.float32
        )

    def __call__(
        self,
        t: Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]],
        t1: Optional[Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]]] = None,
        left: Optional[bool] = True,
    ) -> Float[torch.Tensor, "batch ..."]:
        del left
        if t1 is not None:
            return self(t1) - self(t)

        t = linear_rescale(self.t0, t, self.t1)
        t_polynomial = torch.pow(
            t * torch.ones((5,), device=t.device, dtype=t.dtype),
            exponent=torch.flip(
                torch.arange(5, device=t.device), dims=(0,)
            ),  # pyright : ignore
        )
        return torch.einsum("c, bcf -> bf", t_polynomial, self.coeffs)

    def __repr__(self):
        return (
            f"FourthOrderPolynomialInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coeffs})"
        )
