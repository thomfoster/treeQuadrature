from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class UniformSampler(Sampler):
    """
    Uniform sampler in a hyper-rectangle
    """

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray):
        """
        Argument
        --------
        n : int 
            number of samples
        problem: Problem
            the integration problem being solved
        
        Return
        ------
        np.ndarray of shape (n, self.D)
            samples from the distribution
        """
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        return np.random.uniform(
            low=mins, high=maxs, size=(
                n, D))