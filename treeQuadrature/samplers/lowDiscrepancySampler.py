from .sampler import Sampler
from ..exampleProblems import Problem

from scipy.stats.qmc import Sobol
import numpy as np


class SobolSampler(Sampler):
    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
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
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        n_adjusted = 2 ** np.floor(np.log2(n)).astype(int)
        sampler = Sobol(d=D, scramble=False)
        samples = sampler.random(n_adjusted)
        
        # Map samples from [0, 1] to the integration domain
        samples = samples * (maxs - mins) + mins
        
        return samples
