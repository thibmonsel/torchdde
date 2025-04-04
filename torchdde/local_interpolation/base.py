from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from jaxtyping import Float


class AbstractLocalInterpolation(ABC):
    """Abstract class for creating new interpolation classes."""

    def __init__(
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
        self,
        t: Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]],
        t1: Optional[Union[Float[torch.Tensor, " 1"], Float[torch.Tensor, ""]]] = None,
        left: Optional[bool] = True,
    ) -> Float[torch.Tensor, "batch ..."]:
        """
        Call method for the interpolation class that is used to
        evaluate the interpolator at a given point `t`.
        """
        pass
