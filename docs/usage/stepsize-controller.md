
In order to adjust the time stepping during integrating we provide [`torchdde.AdaptiveStepSizeController`][] for adaptive methods (like [`torchdde.Dopri5`][]) and [`torchdde.ConstantStepSizeController`][] for constant stepsive methods (like [`torchdde.RK4`][]).


::: torchdde.ConstantStepSizeController
    selection:
        members: False

::: torchdde.AdaptiveStepSizeController
    selection:
        members: 
            - __init__
