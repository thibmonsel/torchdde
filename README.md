<h1 align='center'>torchdde</h1>
<!-- <h2 align='center'> Constant lag delay differential equations solver</h2> -->

`torchdde` is a [Pytorch](https://github.com/pytorch/pytorch)-based library providing Constant Lag Delay Differential Equations (DDEs) training neural networks via the adjoint method.

## Installation

```bash
pip install git@github.com:thibmonsel/torchdde.git
```

or local installation

```bash
git clone https://github.com/patrick-kidger/diffrax.git
cd torchdde/
pip install .
```

## Documentation

## Getting started

This example trains a Neural DDE to reproduce a toy dataset from a simple DDE.

### Integrating a simple DDE

```py

```

### Defining

```py

import torch
import torch.nn as nn
from torchvision.ops import MLP

class NDDE(nn.Module):
    def __init__(
        self,
        delays,
        in_size,
        out_size,
        width_size,
        depth,
        activation=nn.ReLU,
        dropout=0,
    ):
        super().__init__()
        self.in_dim = in_size * (1 + len(delays))
        self.delays = torch.nn.Parameter(delays)
        self.mlp = MLP(
            self.in_dim,
            hidden_channels=depth * [width_size] + [out_size],
            activation_layer=activation,
            dropout=dropout,
        )

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.mlp(inp)


```
