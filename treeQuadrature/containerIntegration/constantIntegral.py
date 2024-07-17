import warnings
import numpy as np

from .containerIntegral import ContainerIntegral


class MidpointIntegral(ContainerIntegral):
    '''estimate integral by the function value at mid-point of container'''
    def containerIntegral(self, container, f):
        mid_x = container.midpoint
        mid_y = f(mid_x)[0, 0]
        return mid_y * container.volume


class MedianIntegral(ContainerIntegral):
    '''estimate integral by the median of samples in the container. '''
    def containerIntegral(self, container, f):
        
        if container.N == 0:
            warnings.warn(
                'Attempted to use medianIntegral on Container object with 0' +
                'samples.')
            return 0

        fs = np.array([f(x) for x in container.X])
        median = np.median(fs)

        return median * container.volume