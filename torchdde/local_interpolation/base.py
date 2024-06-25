from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from jaxtyping import Float


class AbstractInterpolation(ABC):
    """Abstract class for creating new interpolation classes."""

    @abstractmethod
    def init(
        self,
        t0: Float[torch.Tensor, ""],
        t1: Float[torch.Tensor, ""],
        dense_info: Dict[str, Float[torch.Tensor, "nb_stages batch ..."]],
        *args: Any,
    ):
        """
        Init method for the interpolation class that is used to compute
        the coefficients of the interpolation polynomial function.
        """
        pass

    @abstractmethod
    def __call__(
        self, t: Float[torch.Tensor, ""], left: Optional[bool] = True
    ) -> Float[torch.Tensor, "batch ..."]:
        """
        Call method for the interpolation class that is used to
        evaluate the interpolator at a given point `t`.
        """
        pass
