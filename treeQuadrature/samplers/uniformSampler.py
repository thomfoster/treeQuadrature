from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np
from typing import Tuple


class UniformSampler(Sampler):
    """
    Uniform sampler in a hyper-rectangle
    """

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable,
            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Argument
        --------
        n : int 
            number of samples
        mins, maxs : np.ndarray
            1 dimensional arrays of the lower bounds
            and upper bounds
        f : function
            the integrand
        
        Return
        ------
        np.ndarray of shape (n, self.D)
            samples from the distribution
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {n}")

        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        xs = np.random.uniform(
            low=mins, high=maxs, size=(
                n, D))

        return xs, f(xs)