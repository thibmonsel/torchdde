# Numerical Solvers

## ODE Solvers

Only a few explicit solvers are available to use :

!!! warning

    The following solver are constant step size solvers. This is indeed less flexible than adaptive stepsize method but such an incorporation isn't available at the time.

::: torchdde.AbstractOdeSolver
    selection:
        members:
            - step
            - integrate

### Explicit Solvers

::: torchdde.Euler
    selection:
        members: false
::: torchdde.Ralston
    selection:
        members: false
::: torchdde.RK2
    selection:
        members: false

::: torchdde.RK4
    selection:
        members: false

### Implicit Solvers

::: torchdde.ImplicitEuler
    selection:
        members: false

## DDE Solvers

::: torchdde.DDESolver
    selection:
        members:
            - __init__
            - step
            - integrate
