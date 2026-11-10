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

    def _find_interval(self, t, left: bool):
        maxlen = self.ts.shape[0] - 2
        index = torch.searchsorted(self.ts, t, side="left" if left else "right")
        return torch.clip(index - 1, 0, maxlen)

    def _interpret_t(
        self, t: Union[Float[torch.Tensor, ""], float], left: bool
    ) -> tuple[Integer[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        index = self._find_interval(t, left)
        # Will never access the final element of `ts`; this is correct behaviour.
        prev_t = self.ts.index_select(0, index.reshape(-1)).reshape(index.shape)
        fractional_part = t - prev_t
        return index, fractional_part

    def __call__(
        self, t: Union[Float[torch.Tensor, ""], float], left=True
    ) -> Float[torch.Tensor, " batch ..."]:
        if self.ys.shape[1] == 1:
            return self.ys[:, 0]
        if t > self.ts[-1] or t < self.ts[0]:
            if torch.abs((t - self.ts[0])) < tiny:
                return self.ys[:, 0]
            if torch.abs((t - self.ts[-1])) < tiny:
                return self.ys[:, -1]
            raise ValueError(
                "Interpolation point is outside data range. "
                + f"ie t={t} > ts[-1]={self.ts[-1]} or t < ts[0]={self.ts[0]}"
            )

        index = self._find_interval(t, left)
        # Keep query-dependent indices on the device: scalar tensor indexing
        # implicitly extracts a Python integer and synchronizes CUDA.
        indices = torch.stack((index, index + 1)).reshape(-1)
        prev_t, next_t = self.ts.index_select(0, indices).reshape(
            2, *index.shape
        ).unbind(0)
        prev_ys, next_ys = self.ys.index_select(1, indices).reshape(
            self.ys.shape[0], 2, *index.shape, *self.ys.shape[2:]
        ).unbind(1)
        fractional_part = t - prev_t
        diff_t = next_t - prev_t
        return prev_ys + (next_ys - prev_ys) * (fractional_part / diff_t)

    def evaluate_many(self, times: Float[torch.Tensor, " query"], left: bool = True):
        """Evaluate a nonempty time vector, returning [query, batch, ...].

        Uses the same endpoint tolerance and left/right selection as scalar
        queries. In-range batches use one range check; endpoint-tolerance checks
        are only needed when a query lies outside the interpolation interval.
        """
        if times.ndim != 1 or times.numel() == 0:
            raise ValueError("Query times must be a nonempty one-dimensional tensor")
        if getattr(self.__call__, "__func__", None) is not TorchLinearInterpolator.__call__:
            return torch.stack([self(t, left=left) for t in times.unbind()])
        first = self.ys[:, 0].unsqueeze(0)
        if self.ys.shape[1] == 1:
            return first.expand(times.numel(), *self.ys.shape[:1], *self.ys.shape[2:])

        outside = (times < self.ts[0]) | (times > self.ts[-1])
        mask_shape = (times.numel(), *([1] * (self.ys.ndim - 1)))
        endpoints = None
        safe_times = times
        if bool(outside.any()):
            near_first = (times - self.ts[0]).abs() < tiny
            near_last = (times - self.ts[-1]).abs() < tiny
            invalid, has_inside = torch.stack((
                (outside & ~(near_first | near_last)).any(), (~outside).any(),
            )).tolist()
            if invalid:
                raise ValueError("Interpolation point is outside data range.")
            # Match scalar precedence when a short interval is close to both ends.
            endpoints = torch.where(
                near_first.reshape(mask_shape), first, self.ys[:, -1].unsqueeze(0),
            )
            if not has_inside:
                return endpoints
            # Keep unused candidates finite, then return exact endpoints below.
            safe_times = torch.where(outside, self.ts[0], times)
        index = self._find_interval(safe_times, left)
        indices = torch.stack((index, index + 1)).reshape(-1)
        prev_t, next_t = self.ts.index_select(0, indices).reshape(2, -1).unbind(0)
        prev_y, next_y = self.ys.index_select(1, indices).reshape(
            self.ys.shape[0], 2, times.numel(), *self.ys.shape[2:],
        ).unbind(1)
        prev_y, next_y = prev_y.movedim(1, 0), next_y.movedim(1, 0)
        fraction = ((safe_times - prev_t) / (next_t - prev_t)).reshape(mask_shape)
        values = prev_y + (next_y - prev_y) * fraction
        return values if endpoints is None else torch.where(
            outside.reshape(mask_shape), endpoints, values,
        )

    def add_point(
        self,
        new_t: Float[torch.Tensor, ""],
        new_y: Float[torch.Tensor, "batch ..."],
    ) -> None:
        if new_y.shape != (self.ys.shape[0], *self.ys.shape[2:]):
            raise ValueError(
                "You tried to add a new value that doesn't fit self.ys's shape."
            )
        index = int(torch.searchsorted(self.ts, new_t))
        replace = index < self.ts.shape[0] and bool(self.ts[index] == new_t)
        end = index + 1 if replace else index
        new_y = new_y.unsqueeze(1)
        new_t = new_t.reshape(1)
        self.ys = torch.cat((self.ys[:, :index], new_y, self.ys[:, end:]), dim=1)
        self.ts = torch.cat((self.ts[:index], new_t, self.ts[end:]))


TorchLinearInterpolator.__init__.__doc__ = """**Arguments:**

- `ts`: Some increasing collection of times. 
- `ys`: The observations themselves.

"""


TorchLinearInterpolator.__call__.__doc__ = """**Arguments:**

- `t`: time to evalute to 
"""

TorchLinearInterpolator.add_point.__doc__ = """**Arguments:**

Insert a point in time order, or replace the value at an existing timestamp.
Replacement keeps timestamps strictly increasing and does not emit a warning.

- `new_t`: new timestamp added to the interpolation
- `new_ys`: new observation added to the interpolation 
"""
