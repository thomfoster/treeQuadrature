from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np


class StratifiedSampler(Sampler):
    def __init__(self, strata_per_dim: int = None, sampling_method: str = 'midpoint'):
        """
        Initialize the StratifiedSampler with user-defined parameters.

        Parameters
        ----------
        strata_per_dim : int, optional
            The number of strata (subdivisions) per dimension. If None, the number of strata will be
            automatically determined based on the number of samples.
        sampling_method : str, optional
            The method to use for sampling within each stratum. Options include:
            - 'midpoint': Sample at the midpoint of each stratum.
            - 'random': Sample randomly within each stratum.
        """
        self.strata_per_dim = strata_per_dim
        self.sampling_method = sampling_method

    def rvs(self, n: int, problem: Problem, *args, **kwargs) -> np.ndarray:
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
        # Determine the number of strata per dimension if not provided
        if self.strata_per_dim is None:
            strata_per_dim = int(np.ceil(n ** (1/problem.D)))
        else:
            strata_per_dim = self.strata_per_dim
        
        # Calculate the total number of strata
        total_strata = strata_per_dim ** problem.D
        if n < total_strata:
            raise ValueError("Number of samples must be greater than or equal to the total number of strata.")

        # Divide each dimension into strata
        stratified_samples = []
        for low, high in zip(problem.lows, problem.highs):
            stratified_intervals = np.linspace(low, high, strata_per_dim + 1)
            if self.sampling_method == 'midpoint':
                stratum_samples = (stratified_intervals[:-1] + stratified_intervals[1:]) / 2.0
            elif self.sampling_method == 'random':
                stratum_samples = stratified_intervals[:-1] + np.random.rand(strata_per_dim) * (stratified_intervals[1:] - stratified_intervals[:-1])
            else:
                raise ValueError("Unsupported sampling method. Use 'midpoint' or 'random'.")
            stratified_samples.append(stratum_samples)
        
        # Create a meshgrid of the strata
        meshgrid = np.meshgrid(*stratified_samples)
        grid_points = np.vstack([mg.flatten() for mg in meshgrid]).T
        
        # Select samples randomly from the grid points
        selected_indices = np.random.choice(grid_points.shape[0], n, replace=True)
        
        return grid_points[selected_indices]