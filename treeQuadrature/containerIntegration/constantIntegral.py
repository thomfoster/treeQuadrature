import warnings
import numpy as np

from .containerIntegral import ContainerIntegral


class MidpointIntegral(ContainerIntegral):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'MidpointIntegral'

    '''estimate integral by the function value at mid-point of container'''
    def containerIntegral(self, container, f):
        mid_x = container.midpoint
        mid_y = f(mid_x)[0, 0]
        return mid_y * container.volume


class MedianIntegral(ContainerIntegral):
    '''estimate integral by the median of samples in the container. '''
    def __init__(self, return_std: bool=False):
        self.return_std = return_std
        self.name = 'MedianIntegral'

    def containerIntegral(self, container, f, return_std: bool=False):
        if container.N == 0:
            warnings.warn(
                'Attempted to use medianIntegral on Container object with 0' +
                'samples.', RuntimeWarning)
            return (0, 0) if return_std else 0

        fs = np.array([f(x) for x in container.X])
        median = np.median(fs)
        integral_estimate = median * container.volume

        if return_std or self.return_std:
            std_estimate = np.std(fs * container.volume) / np.sqrt(container.N)
            return (integral_estimate, std_estimate)
        
        return integral_estimate