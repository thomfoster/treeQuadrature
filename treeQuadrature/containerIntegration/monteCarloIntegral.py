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
    n_samples : int
        number of samples to be redrawn for evaluating the integral
        default : 10
    eval : function
        Should take a np.ndarray (samples of integrand) 
        and return a float (aggregated value) 
        Default is mean 
    error_estimate : function
        must be provided when eval is not mean nor median
        obtain an estimate of the uncertainty
    sampler : Sampler
        The sampler used to draw samples in the container
        Default is UniformSampler
    '''
    def __init__(self, n_samples: int = 10, eval: Callable=np.mean, 
                 error_estimate: Optional[Callable]=None, 
                 sampler: Sampler=defaultSampler) -> None:
        self.n_samples = n_samples
        self.name = 'RandomIntegral'
        self.eval = eval  
        self.error_estimate = error_estimate
        self.sampler = sampler

    def containerIntegral(self, container: Container, f: Callable, 
                          return_std: bool=False):

        xs, ys = self.sampler.rvs(self.n_samples, container.mins, 
                                   container.maxs, f)
        container.add(xs, ys)  # for tracking num function evaluations
        # ignore previous samples in the container 
        # which will give skewed estimates
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