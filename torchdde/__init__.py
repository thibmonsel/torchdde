import importlib.metadata

from .global_interpolation import (
    DenseInterpolation as DenseInterpolation,
    TorchLinearInterpolator as TorchLinearInterpolator,
)
from .integrate import integrate as integrate
from .local_interpolation import (
    AbstractInterpolation as AbstractInterpolation,
    FirstOrderPolynomialInterpolation as FirstOrderPolynomialInterpolation,
    FourthOrderPolynomialInterpolation as FourthOrderPolynomialInterpolation,
    ThirdOrderPolynomialInterpolation as ThirdOrderPolynomialInterpolation,
)
from .solver import (
    AbstractOdeSolver as AbstractOdeSolver,
    Dopri5 as Dopri5,
    Euler as Euler,
    ImplicitEuler as ImplicitEuler,
    RK2 as RK2,
    RK4 as RK4,
)
from .step_size_controller import (
    AbstractStepSizeController as AbstractStepSizeController,
    AdaptiveStepSizeController as AdaptiveStepSizeController,
    ConstantStepSizeController as ConstantStepSizeController,
)


__version__ = importlib.metadata.version("torchdde")
