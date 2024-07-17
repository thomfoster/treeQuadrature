import numpy as np

from .integrator import Integrator
from ..exampleProblems import Problem

class SmcIntegrator(Integrator):
    """
    Simple integrator: Draw N samples from prior,
    take sample mean of likelihood values at these samples.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    """
    def __init__(self, N: int):
        self.N = N

    def __call__(self, problem: Problem, return_N: bool=False):
        """
        Perform the integration process.

        Parameters
        ----------
        problem : Problem
            The integration problem
        return_N : bool, optional
            If True, return the number of samples used.

        Returns
        -------
        result : float or tuple
            The computed integral and optionally the number of samples
            and other details
        """
        # Draw N samples from the prior distribution
        xs = problem.p.rvs(self.N)
        # Evaluate the likelihood at these samples
        ys = problem.d.pdf(xs).reshape(-1)
        G = np.mean(ys)

        ret = {'estimate': G}
        if return_N:
            ret['n_evals'] = self.N
        return ret
