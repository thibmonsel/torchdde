import numpy as np
import torch


class TorchLinearInterpolator():

    def __init__(self,ts, ys, device):

        self.ts = ts.to(device) # [N_t]
        self.ys = ys.to(device) # [N, N_t, D]

    def __post_init__(self):
        if self.ts.ndim != 1:
            raise ValueError("`ts` must be one dimensional.")
        
        if self.ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                    "number of entries along the timelike dimension."
                )
        
        if np.all(np.diff(self.ts) > 0) :
            raise ValueError("`ts` must be monotonically increasing.")
                
    def _interpret_t(self, t : float, left: bool):
        maxlen = self.ts.shape[0] - 2
        index = torch.searchsorted(self.ts, t, side="left" if left else "right")
        index = torch.clip(index - 1, 0, maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part
    
    def __call__(self, t, left=True):
        
        if t > self.ts[-1] or t < self.ts[0]:
            raise ValueError("Interpolation point is outside data range. ie t > ts[-1] or t < ts[0]")
        index, fractional_part = self._interpret_t(t, left)
        prev_ys = self.ys[index]
        next_ys = self.ys[index + 1]
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t

        return prev_ys + (next_ys - prev_ys) * (fractional_part / diff_t)




