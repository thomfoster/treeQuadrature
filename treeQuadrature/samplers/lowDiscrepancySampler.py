from .sampler import Sampler

from scipy.stats.qmc import Sobol
import numpy as np
from typing import Tuple


class SobolSampler(Sampler):
    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable,
            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Low-discrepancy sampling using Sobol sequences.

        Parameters
        ----------
        n : int
            Number of samples.
        mins, maxs : np.ndarray
            1 dimensional arrays of the lower bounds
            and upper bounds
        f : function
            the integrand
        
        Returns
        -------
        np.ndarray
            Samples from the distribution.
        """
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        n_adjusted = 2 ** np.floor(np.log2(n)).astype(int)
        sampler = Sobol(d=D, scramble=False)
        xs = sampler.random(n_adjusted)
        
        # Map xs from [0, 1] to the integration domain
        xs = xs * (maxs - mins) + mins
        
        return xs, f(xs)
