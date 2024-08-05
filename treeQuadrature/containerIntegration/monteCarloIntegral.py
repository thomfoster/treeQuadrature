import numpy as np

from .containerIntegral import ContainerIntegral


class RandomIntegral(ContainerIntegral):
    '''
    Monte Carlo integrator:
    redraw samples, and estimate integral by the median

    Parameters
    ----------
    n : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    '''
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.name = 'RandomIntegral'

    def containerIntegral(self, container, f, **kwargs):
        n = kwargs.get('n', self.n)

        samples = container.rvs(n)
        ys = f(samples)
        container.add(samples, ys)  # for tracking num function evaluations
        # I deliberately ignore previous samples which give skewed estimates
        y = np.median(ys)

        integral_estimate = y * container.volume
        return {'integral' : integral_estimate}


class SmcIntegral(ContainerIntegral):
    """
    Monte Carlo integrator:
    redraw samples, and estimate integral by the mean

    Attributes
    ----------
    n : int, optional
        number of samples to be redrawn for evaluating the integral
        default : 10
    """
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.name = 'SmcIntegral'

    def containerIntegral(self, container, f, **kwargs):
        n = kwargs.get('n', self.n)

        samples = container.rvs(n)
        ys = f(samples)
        container.add(samples, ys)  # for tracking num function evaluations
        v = container.volume

        integral_estimate = v * np.mean(ys)
        return {'integral' : integral_estimate}
