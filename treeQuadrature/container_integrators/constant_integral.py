import warnings
import numpy as np

from .container_integral import ContainerIntegral
from ..container import Container


class MidpointIntegral(ContainerIntegral):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'MidpointIntegral'

    '''estimate integral by the function value at mid-point of container'''
    def containerIntegral(self, container: Container, f: callable):
        mid_x = container.midpoint
        mid_y = f(mid_x)[0, 0]
        integral_estimate = mid_y * container.volume
        
        return {'integral' : integral_estimate}


class MedianIntegral(ContainerIntegral):
    '''estimate integral by the median of samples in the container. '''
    def __init__(self):
        self.name = 'MedianIntegral'

    def containerIntegral(self, container: Container, f: callable, 
                          return_std: bool=False):
        if container.N == 0:
            warnings.warn(
                'Attempted to use medianIntegral on Container object with 0' +
                'samples.', RuntimeWarning)
            return {'integral': 0.0, 'std': 0.0} if return_std else {'integral': 0.0}

        fs = np.array([f(x) for x in container.X])
        median = np.median(fs)
        integral_estimate = median * container.volume

        ret = {'integral' : integral_estimate}

        if return_std:
            std_estimate = np.std(fs * container.volume) / np.sqrt(container.N)
            ret['std'] = std_estimate
        
        return ret
