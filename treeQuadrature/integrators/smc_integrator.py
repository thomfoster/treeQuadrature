import numpy as np
from typing import Optional

from .integrator import Integrator
from ..example_problems import Problem, BayesProblem
from ..samplers import Sampler
from ..utils import ResultDict


class SmcIntegrator(Integrator):
    """
    Simple Monte Carlo integrator: Draw N samples from prior,
    take sample mean of likelihood values at these samples. \n
    Only works for BayesProblem

    Parameters
    ----------
    N : int
        Number of samples to draw.
    sampler
    """
    def __init__(self, N: int, sampler: Optional[Sampler]=None):
        self.N = N
        self.sampler = sampler

    def __call__(self, problem: Problem, return_N: bool=False, 
                 return_std: bool=False) -> ResultDict:
        """
        Perform the integration process.

        Parameters
        ----------
        problem : Problem
            The integration problem
        return_N : bool, optional
            If True, return the number of samples used.
        return_std : bool, optional
            If True, return the standard deviation of the Monte Carlo estimate.

        Return
        -------
        dict
            with the following keys:
            - 'estimate' (float) : estimated integral value
            - 'n_evals' (int) :  number of function estiamtions, if return_N is True
            - 'std' (float) : standard deviation of the estimate, if return_std is True
        """
        # Draw N samples from the prior distribution
        if isinstance(problem, BayesProblem):
            xs = problem.p.rvs(self.N)
            ys = problem.integrand(xs)
        else:
            raise ValueError('problem is not BayesProblem, and '
                             'integrator does not have sampler')
        # Evaluate the likelihood at these samples
        G = np.mean(ys)
        std_G = np.std(ys) / np.sqrt(self.N)  # Standard deviation of the mean

        ret = ResultDict(estimate=G)
        if return_N:
            ret['n_evals'] = self.N
        if return_std:
            ret['std'] = std_G
        return ret
