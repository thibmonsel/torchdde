from typing import Dict, Optional, Union

import torch
from jaxtyping import Float

from torchdde.local_interpolation.base import AbstractLocalInterpolation


class FirstOrderPolynomialInterpolation(AbstractLocalInterpolation):
    cache_coefficients = True
    vectorized_evaluation = True

    @staticmethod
    def evaluate_many(t0, t1, t, coefficients):
        slope, y0, _ = coefficients.unbind(dim=2)
        delta = (t - t0).reshape(t.shape[0], *([1] * (slope.ndim - 1)))
        return torch.addcmul(y0, slope, delta)

    def __init__(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "batch ..."]],
    ):
        self.t0 = t0
        self.t1 = t1
        if "_cached_coefficients" in dense_info:
            self.coeffs = dense_info["_cached_coefficients"]
            self.slope, self.y0, self.y1 = self.coeffs.unbind(dim=1)
        else:
            self.y0 = dense_info["y0"]
            self.y1 = dense_info["y1"]
            dt = self.t1 - self.t0
            dt = torch.where(dt.abs() > 0.0, dt, 1.0)
            self.slope = (self.y1 - self.y0) / dt
            self.coeffs = torch.stack((self.slope, self.y0, self.y1), dim=1)

    def __call__(
        self,
        t: Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]],
        t1: Optional[Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]]] = None,
        left: Optional[bool] = True,
    ) -> Float[torch.Tensor, "batch ..."]:
        del t1, left
        return torch.addcmul(self.y0, self.slope, t - self.t0)
