from typing import Dict, Union

import torch
from jaxtyping import Float, Integer


class DenseInterpolation:
    def __init__(
        self,
        direction,
        interpolation_cls,
        ts: Float[torch.Tensor, " 2"],
        infos: Dict[str, Float[torch.Tensor, "1 batch ..."]],
    ):
        self.direction = direction
        self.ts = ts
        self.infos = infos
        self.interpolation_cls = interpolation_cls

    def __check_init__(self):
        if self.ts is not None and self.infos is not None:
            assert (
                self.ts.shape[0] == self.infos[list(self.infos.keys())[0]].shape[0] + 1
            )

    def _interpret_t(
        self, t: Union[Float[torch.Tensor, ""], float], left: bool
    ) -> tuple[Integer[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        maxlen = self.ts.shape[0] - 2
        index = torch.searchsorted(self.ts, t, side="left" if left else "right")
        index = torch.clip(index - 1, 0, maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    def _get_local_interpolation(self, t, left: bool):
        index, _ = self._interpret_t(t, left)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        dense_info = []
        for elmt in self.infos.values():
            dense_info.append(elmt[index])
        specific_infos = dict(zip(self.infos.keys(), dense_info))
        return self.interpolation_cls(t0=prev_t, t1=next_t, dense_info=specific_infos)

    def __call__(self, t0, t1=None, left: bool = True):
        if t1 is not None:
            return self.__call__(t1, left=left) - self.__call__(t0, left=left)
        t = t0 * self.direction
        out = self._get_local_interpolation(t, left).__call__(t, left=left)
        return out

    def add_point(self, t, dense_info):
        if self.direction:
            self.ts = torch.cat([self.ts, t])
            for key in dense_info.keys():
                self.infos[key] = torch.concat(
                    [self.infos[key], dense_info[key].unsqueeze(0)], dim=0
                )
        else:
            self.ts = torch.cat([t, self.ts])
            for key in dense_info.keys():
                self.infos[key] = torch.concat(
                    [dense_info[key].unsqueeze(0), self.infos[key]], dim=0
                )

        self.__check_init__()
