
In order to adjust the time stepping during integrating we provide `torchdde.AdaptiveStepSizeController` for Adaptive methods (like [`torchdde.Dopri5`][]) and `torchdde.ConstantStepSizeController` for constant stepsive methods (like [`torchdde.RK4`][]).

::: torchdde.AbstractStepSizeController
    selection:
        members:
            - init
            - adapt_step_size

::: torchdde.ConstantStepSizeController
    selection:
        members: False

::: torchdde.AdaptiveStepSizeController
    selection:
        members: 
            - __init__
