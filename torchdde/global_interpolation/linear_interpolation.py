import warnings
from typing import Union

import torch
from jaxtyping import Float, Integer


tiny = 10e-3


class TorchLinearInterpolator:
    r"""Linear interpolator class that is compatible with batching.
    All elements in batch must have the same ts."""

    def __init__(
        self,
        ts: Float[torch.Tensor, " time"],
        ys: Float[torch.Tensor, "batch time ..."],
    ) -> None:
        if ts.ndim != 1:
            raise ValueError("`ts` must be one dimensional.")

        if ys.shape[1] != ts.shape[0]:
            raise ValueError(
                "Must have ts.shape[0] == ys.shape[1], that is to say the same "
                "number of entries along the timelike dimension."
            )

        if not torch.all(torch.diff(ts) > 0):
            raise ValueError("`ts` must be monotonically increasing.")

        self.ts = ts
        self.ys = ys

    def to(self, device):
        self.ys = self.ys.to(device)
        self.ts = self.ts.to(device)

    def _interpret_t(
        self, t: Union[Float[torch.Tensor, ""], float], left: bool
    ) -> tuple[Integer[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        maxlen = self.ts.shape[0] - 2
        index = torch.searchsorted(self.ts, t, side="left" if left else "right")
        index = torch.clip(index - 1, 0, maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    def __call__(
        self, t: Union[Float[torch.Tensor, ""], float], left=True
    ) -> Float[torch.Tensor, " batch ..."]:
        if t > self.ts[-1] or t < self.ts[0]:
            if self.ys.shape[1] == 1:
                return self.ys[:, 0]
            if torch.abs((t - self.ts[0])) < tiny:
                return self.ys[:, 0]
            if torch.abs((t - self.ts[-1])) < tiny:
                return self.ys[:, -1]
            raise ValueError(
                "Interpolation point is outside data range. "
                + f"ie t={t} > ts[-1]={self.ts[-1]} or t < ts[0]={self.ts[0]}"
            )

        index, fractional_part = self._interpret_t(t, left)
        prev_ys = self.ys[:, index]
        next_ys = self.ys[:, index + 1]
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t
        return prev_ys + (next_ys - prev_ys) * (fractional_part / diff_t)

    def add_point(
        self,
        new_t: Float[torch.Tensor, ""],
        new_y: Float[torch.Tensor, "batch ..."],
    ) -> None:
        if new_t in self.ts:
            warnings.warn(
                f"already have new_t={new_t} point in interpolation, overwriting it"
            )

        new_y = torch.unsqueeze(new_y, dim=1)
        new_t = torch.unsqueeze(new_t.clone(), dim=0)

        if self.ys.shape[-1] != new_y.shape[-1]:
            raise ValueError(
                "You tried to add a new value that doesn't fit self.ys's shape."
            )
        rel_position = self.ts < new_t
        last_insertion = torch.sum(rel_position)
        if last_insertion == len(rel_position):
            new_ys = torch.concat((self.ys, new_y), dim=1)
            new_ts = torch.concat((self.ts, new_t))
        elif last_insertion == 0:
            new_ys = torch.concat((new_y, self.ys), dim=1)
            new_ts = torch.concat((new_t, self.ts))
        else:
            index = rel_position.nonzero()[-1] - 1
            new_ys = torch.concat(
                (self.ys[:, :index], new_y, self.ys[:, index:]), dim=1
            )
            new_ts = torch.concat((self.ts[:index], new_t, self.ts[index:]))

        self.ys = new_ys
        self.ts = new_ts


TorchLinearInterpolator.__init__.__doc__ = """**Arguments:**

- `ts`: Some increasing collection of times. 
- `ys`: The observations themselves.

"""


TorchLinearInterpolator.__call__.__doc__ = """**Arguments:**

- `t`: time to evalute to 
"""

TorchLinearInterpolator.add_point.__doc__ = """**Arguments:**

- `new_t`: new timestamp added to the interpolation
- `new_ys`: new observation added to the interpolation 
"""
