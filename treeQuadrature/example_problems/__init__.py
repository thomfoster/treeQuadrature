from .base_class import Problem
from .bayes_problems import (
    BayesProblem,
    SimpleGaussian,
    Camel,
    QuadCamel,
    Gaussian
)
from .simple_problems import (
    Pyramid,
    Quadratic,
    ExponentialProduct,
    ProductPeak,
    CornerPeak,
    C0,
    Discontinuous,
)
from .complex_problems import Ripple, Oscillatory

__all__ = [
    "Problem",
    "BayesProblem",
    "SimpleGaussian",
    "Camel",
    "QuadCamel",
    "Gaussian",
    "Pyramid",
    "Quadratic",
    "ExponentialProduct",
    "ProductPeak",
    "CornerPeak",
    "C0",
    "Discontinuous",
    "Ripple",
    "Oscillatory",
]
