import numpy as np
from typing import Callable, Optional

from .containerIntegral import ContainerIntegral
from ..container import Container
from ..samplers import Sampler, UniformSampler


defaultSampler = UniformSampler()

class RandomIntegral(ContainerIntegral):
    '''
    Monte Carlo integrator:
    redraw samples, and estimate integrand 
    by the a constant value

    Parameters
    ----------
    n : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    eval : function, optional
        Should take a np.ndarray (samples of integrand) 
        and return a float (aggregated value) 
        Default is mean 
    error_estimate : function, optional
        must be provided when eval is not mean nor median
        obtain an estimate of the uncertainty
    '''
    def __init__(self, n: int = 10, eval: Callable=np.mean, 
                 error_estimate: Optional[Callable]=None, 
                 sampler: Sampler=defaultSampler) -> None:
        self.n = n
        self.name = 'RandomIntegral'
        self.eval = eval  
        self.error_estimate = error_estimate
        self.sampler = sampler

    def containerIntegral(self, container: Container, f: Callable, 
                          return_std: bool=False):

        samples = container.rvs(self.n)
        ys = f(samples)
        container.add(samples, ys)  # for tracking num function evaluations
        # I deliberately ignore previous samples which give skewed estimates
        y = self.eval(ys)

        integral_estimate = y * container.volume
        results = {'integral' : integral_estimate}

        if return_std:
            std = self.estimate_error(ys)
            results['std'] = std * container.volume

        return results
    
    def estimate_error(self, samples: np.ndarray):
        if len(samples) < 2:
            return 0
        
        if self.eval == np.mean:
            std = np.sqrt(np.var(samples) / len(samples))
        elif self.eval == np.median:
            median_value = np.median(samples)
            mad = np.median(np.abs(samples - median_value))
            std = mad / np.sqrt(len(samples))
        elif self.error_estimate is not None:
            std = self.error_estimate(samples)
        else:
            raise ValueError("evaluation method is not mean nor median, "
                             "must specify 'error_estimate'")
        
        return std