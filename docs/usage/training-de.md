# Training DDEs/ODEs

First, there are a lot of available package to use to train Neural ODEs, [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (not maintained anymore) in Pytorch and [Diffrax](https://github.com/patrick-kidger/diffrax) (which is the gold standard here). This means that this library doesn't have any many features since it focuses more on DDEs.

Two following ways are possible to train Neural DDE / Neural ODE :

- optimize-then-discretize (with the adjoint method)
- discretize-then-optimize (regular backpropagation)

Please see the doctorial thesis [On Neural Differential Equations](https://arxiv.org/pdf/2202.02435.pdf) for more information on both procedures.

Regarless, the only entry point to integrate ODEs/DDEs is the `integrate` function.

::: torchdde.integrate.integrate
    selection:
        members: True

!!! warning

    You are unable to learn the DDE's delays if using the `discretize_then_optimize=True`.
