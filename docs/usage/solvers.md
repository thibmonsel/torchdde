# Numerical Solvers

## ODE Solvers

Only some explicit solvers are available to use but adding new ones is rather simple with [`torchdde.AbstractOdeSolver`][].

::: torchdde.AbstractOdeSolver
    selection:
        members:
            - init
            - order
            - step
            - build_interpolation

### Explicit Solvers

::: torchdde.Euler
    selection:
        members: false
::: torchdde.RK2
    selection:
        members: false
::: torchdde.RK4
    selection:
        members: false
::: torchdde.Dopri5
    selection:
        members: false

### Implicit Solvers

::: torchdde.ImplicitEuler
    selection:
        members: false
