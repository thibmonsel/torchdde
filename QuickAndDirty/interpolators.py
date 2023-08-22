import numpy as np
import torch


class TorchLinearInterpolator:
    def __init__(self, ts, ys, device):

        self.ts = ts.to(device)  # [N_t]
        self.ys = ys.to(device)  # [N, N_t, D]
        self.device = device
        
    def __post_init__(self):
        if self.ts.ndim != 1:
            raise ValueError("`ts` must be one dimensional.")

        if self.ys.shape[1] != self.ts.shape[0]:
            raise ValueError(
                "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                "number of entries along the timelike dimension."
            )

        if not torch.all(torch.diff(self.ts) > 0):
            raise ValueError("`ts` must be monotonically increasing.")

    def _interpret_t(self, t: float, left: bool):
        maxlen = self.ts.shape[0] - 2
        index = torch.searchsorted(self.ts, t, side="left" if left else "right")
        index = torch.clip(index - 1, 0, maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    def __call__(self, t, left=True):
        if t > self.ts[-1] or t < self.ts[0]:
            raise ValueError(
                "Interpolation point is outside data range. ie t > ts[-1] or t < ts[0]"
            )
        t = torch.tensor(t)
        t = t.to(self.device)
        index, fractional_part = self._interpret_t(t, left)
        prev_ys = self.ys[:, index]
        next_ys = self.ys[:, index + 1]
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t
        return prev_ys + (next_ys - prev_ys) * (fractional_part / diff_t)

    def add_point(self, new_t, new_y):
        # new_t : float
        # new_y : torch.tensor size [N, D]
        if new_t in self.ts :
            return 
        
        new_y = new_y.to(self.ts.device)
        new_y = torch.unsqueeze(new_y, dim=1)
        new_t = torch.unsqueeze(torch.tensor(new_t), dim=0)
        new_t = new_t.to(self.ts.device)
        if self.ys.shape[-1] != new_y.shape[-1]:
            raise ValueError(
                "You tried to add a new value that doesn't fit the shape of self.ys "
            )

        rel_position = self.ts < new_t

        if torch.all(rel_position) :
            new_ys = torch.concat((self.ys, new_y), dim=1)
            new_ts = torch.concat((self.ts, new_t)) 
        elif not torch.all(rel_position) : 
            new_ys = torch.concat((new_y, self.ys), dim=1)
            new_ts = torch.concat((new_t, self.ts))
        else : 
            index = rel_position.nonzero()[-1] - 1
            new_ys = torch.concat(
                (self.ys[:, :index], new_y, self.ys[:, index:]), dim=1
            )
            new_ts = torch.concat((self.ts[:index], new_t, self.ts[index:]))

        self.ys = new_ys
        self.ts = new_ts

        if not torch.all(torch.diff(self.ts) > 0):
            raise ValueError(
                "`ts` must be monotonically increasing. oups errors in add_point"
            )
