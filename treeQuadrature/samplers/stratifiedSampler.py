from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class StratifiedSampler(Sampler):
    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Stratified sampling to ensure coverage of the entire domain.

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
        samples_per_stratum = int(np.ceil(n ** (1/problem.D)))
        stratified_samples = []
        
        for low, high in zip(problem.lows, problem.highs):
            stratified_intervals = np.linspace(low, high, samples_per_stratum + 1)
            stratum_samples = (stratified_intervals[:-1] + stratified_intervals[1:]) / 2.0
            stratified_samples.append(stratum_samples)
        
        meshgrid = np.meshgrid(*stratified_samples)
        grid_points = np.vstack([mg.flatten() for mg in meshgrid]).T
        selected_indices = np.random.choice(grid_points.shape[0], n, replace=True)
        
        return grid_points[selected_indices]
