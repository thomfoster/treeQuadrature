from .sampler import Sampler
from ..exampleProblems import Problem

import numpy as np
from typing import Tuple


class AdaptiveStratifiedSampler(Sampler):
    def __init__(self, initial_strata_per_dim: int = 5, 
                 refinement_threshold: float = 0.1, 
                 max_refinement_levels: int = 3):
        """
        Initialize the AdaptiveStratifiedSampler.

        Parameters
        ----------
        initial_strata_per_dim : int, optional
            Initial number of strata (subdivisions) per dimension.
        refinement_threshold : float, optional
            Threshold for refining a stratum based on the integrand's value.
        max_refinement_levels : int, optional
            Maximum number of refinement levels.
        """
        self.initial_strata_per_dim = initial_strata_per_dim
        self.refinement_threshold = refinement_threshold
        self.max_refinement_levels = max_refinement_levels

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive stratified sampling to ensure coverage of the entire domain and focus on important regions.

        Parameters
        ----------
        n : int
            Number of samples.
        mins : np.ndarray
            1-dimensional array of the lower bounds of the domain.
        maxs : np.ndarray
            1-dimensional array of the upper bounds of the domain.
        f : callable
            The integrand function to be sampled.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            xs : np.ndarray of shape (n, D)
                The sampled points.
            ys : np.ndarray of shape (n, )
                The integrand values at the sampled points.
        """
        samples = []
        values = []
        strata = [(mins, maxs)]
        D = len(mins)

        for level in range(self.max_refinement_levels):
            new_strata = []
            for low, high in strata:
                # Subdivide each stratum
                sub_strata = self.subdivide_stratum(low, high, self.initial_strata_per_dim)
                for sub_low, sub_high in sub_strata:
                    # Sample within each sub-stratum
                    sub_samples = np.random.uniform(sub_low, sub_high, (n // len(strata), D))
                    sub_values = f(sub_samples)
                    if np.mean(sub_values) > self.refinement_threshold:
                        new_strata.append((sub_low, sub_high))
                    samples.append(sub_samples)
                    values.append(sub_values)
            strata = new_strata

        xs = np.vstack(samples)
        ys = np.concatenate(values)
        
        # If we collected more samples than requested due to rounding, trim them
        if xs.shape[0] > n:
            indices = np.random.choice(xs.shape[0], n, replace=False)
            xs = xs[indices]
            ys = ys[indices]
        
        return xs, ys
    
    def subdivide_stratum(self, low: np.ndarray, high: np.ndarray, strata_per_dim: int) -> list:
        """
        Subdivide a stratum into smaller sub-strata.

        Parameters
        ----------
        low, high : np.ndarray
            Lower and upper bounds of the stratum.
        strata_per_dim : int
            Number of subdivisions per dimension.

        Returns
        -------
        list of tuples
            Subdivided strata as a list of (low, high) tuples.
        """
        sub_strata = []
        for i in range(strata_per_dim):
            for j in range(strata_per_dim):
                sub_low = low + i * (high - low) / strata_per_dim
                sub_high = low + (i + 1) * (high - low) / strata_per_dim
                sub_strata.append((sub_low, sub_high))
        return sub_strata