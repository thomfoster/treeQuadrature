from .base_class import Sampler

from scipy.stats.qmc import Sobol
import numpy as np
from typing import Tuple


class SobolSampler(Sampler):
    def rvs(
        self, n: int, mins: np.ndarray, maxs: np.ndarray, f: callable, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        # Find the next power of two greater than or equal to n
        n_adjusted = 2 ** np.ceil(np.log2(n)).astype(int)
        sampler = Sobol(d=D, scramble=False)

        # Generate more samples than needed if necessary
        xs = sampler.random(n_adjusted)

        # Map xs from [0, 1] to the integration domain
        xs = xs * (maxs - mins) + mins

        # If more samples were generated, truncate to the desired amount
        if n < n_adjusted:
            xs = xs[:n]

        return xs, f(xs)
