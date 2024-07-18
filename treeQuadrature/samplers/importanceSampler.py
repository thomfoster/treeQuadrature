from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class ImportanceSampler(Sampler):
    """
    Importance sampler based on the integrand function
    """

    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Generate importance sampling random samples.

        Parameters
        ----------
        n : int 
            Number of samples.
        problem: Problem
            The integration problem being solved.

        Returns
        -------
        np.ndarray
            Samples from the importance sampling distribution.
        """
        samples = np.random.uniform(low=problem.lows, high=problem.highs, size=(n, problem.D))
        integrand_values = problem.integrand(samples).reshape(-1)
        weights = np.abs(integrand_values)
        probabilities = weights / weights.sum()

        indices = np.random.choice(np.arange(n), size=n, p=probabilities)
        return samples[indices]
