from typing import Dict, Optional, Union

import torch
from jaxtyping import Float


class FirstOrderPolynomialInterpolation:
    def __init__(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "batch ..."]],
    ):
        self.t0 = t0
        self.t1 = t1
        self.y0 = dense_info["y0"]
        self.y1 = dense_info["y1"]

    def __call__(
        self,
        t: Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]],
        left: Optional[bool] = True,
    ) -> Float[torch.Tensor, "batch ..."]:
        dt = self.t1 - self.t0
        dt = torch.where(dt.abs() > 0.0, dt, 1.0)
        coeff = (self.y1 - self.y0) / dt
        return self.y0 + coeff * (t - self.t0)
