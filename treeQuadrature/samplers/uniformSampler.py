from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class UniformSampler(Sampler):
    """
    Uniform sampler in a hyper-rectangle
    """

    def rvs(self, n: int, problem: Problem):
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
        return np.random.uniform(
            low=problem.lows, high=problem.highs, size=(
                n, problem.D))