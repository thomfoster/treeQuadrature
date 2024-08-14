from .sampler import Sampler
from ..exampleProblems import Problem

from scipy.stats.qmc import Sobol
import numpy as np


class LowDiscrepancySampler(Sampler):
    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Low-discrepancy sampling using Sobol sequences.

        Parameters
        ----------
        n : int
            Number of samples.
        problem : Problem
            The integration problem being solved.
        
        Returns
        -------
        np.ndarray
            Samples from the distribution.
        """
        n_adjusted = 2 ** np.ceil(np.log2(n)).astype(int)
        sampler = Sobol(d=problem.D, scramble=False)
        samples = sampler.random(n_adjusted)
        
        # Map samples from [0, 1] to the integration domain
        samples = samples * (problem.highs - problem.lows) + problem.lows
        
        return samples
