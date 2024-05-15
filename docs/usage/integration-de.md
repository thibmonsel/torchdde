# Integration

First, there are a lot of available package to use to train/integrate Neural ODEs, [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (not maintained anymore) in Pytorch and [Diffrax](https://github.com/patrick-kidger/diffrax) (which is the gold standard here). This means that this library doesn't have any many features since it focuses more on DDEs.

Regarless, the only entry point to integrate ODEs/DDEs is the `integrate` function.

What essentially differentiates a DDE with an ODE are :

- the vector field definition.
- the `delays` argument specification.  

??? tip "What changes in an ODE compared to a DDE ?"

    In practice, your function will be defined like this :  
    ```python
    def f_ode(t,y,args):
        return ...
    
    def f_dde(t,y,args, history):
        return ...
    ```

    and your ODE's initial condition `y0` will become a history function `history_fn = lambda t : ...`


::: torchdde.integrate.integrate
    selection:
        members: True