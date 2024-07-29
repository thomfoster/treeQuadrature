import numpy as np

from .integrator import Integrator
from ..exampleProblems import BayesProblem

class SmcIntegrator(Integrator):
    """
    Simple integrator: Draw N samples from prior,
    take sample mean of likelihood values at these samples.
    only works for BayesProblem

    Parameters
    ----------
    N : int
        Number of samples to draw.
    """
    def __init__(self, N: int):
        self.N = N

    def __call__(self, problem: BayesProblem, return_N: bool=False, 
                 return_std: bool=False):
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
        if problem.p is not None:
            xs = problem.p.rvs(self.N)
        else:
            xs = problem.rvs(self.N)
        # Evaluate the likelihood at these samples
        ys = problem.d.pdf(xs).reshape(-1)
        G = np.mean(ys)
        std_G = np.std(ys) / np.sqrt(self.N)  # Standard deviation of the mean

        ret = {'estimate': G}
        if return_N:
            ret['n_evals'] = self.N
        if return_std:
            ret['std'] = std_G
        return ret
