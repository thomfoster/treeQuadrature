from .base_class import Sampler

import numpy as np
from typing import Tuple, Callable


class StratifiedSampler(Sampler):
    def __init__(self, strata_per_dim: int = 4,
                 sampling_method: str = "midpoint"):
        """
        Initialize the StratifiedSampler with user-defined parameters.

        Parameters
        ----------
        strata_per_dim : int, optional
            The number of strata (subdivisions) per dimension.
            Default is 4.
        sampling_method : str, optional
            The method to use for sampling within each stratum. \n
            Options include:
            - 'midpoint': Sample at the midpoint of each stratum.
            - 'random': Sample randomly within each stratum.
        """
        self.strata_per_dim = strata_per_dim
        self.sampling_method = sampling_method

    def rvs(
        self, n: int, mins: np.ndarray, maxs: np.ndarray, f: Callable, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified sampling to ensure coverage of the entire domain.

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

        max_strata_per_dim = max(1, int(np.floor(n ** (1 / D))))
        self.strata_per_dim = min(self.strata_per_dim, max_strata_per_dim)

        total_strata = self.strata_per_dim**D
        if n < total_strata:
            raise ValueError(
                f"Number of samples ({n}) must be greater than or equal to "
                f"the total number of strata ({total_strata})."
            )

        stratified_samples = []
        for low, high in zip(mins, maxs):
            stratified_intervals = np.linspace(
                low, high, self.strata_per_dim + 1)
            if self.sampling_method == "midpoint":
                stratum_samples = (
                    stratified_intervals[:-1] + stratified_intervals[1:]
                ) / 2.0
            elif self.sampling_method == "random":
                stratum_samples = stratified_intervals[:-1] + (
                    np.random.rand(self.strata_per_dim)
                    * (stratified_intervals[1:] - stratified_intervals[:-1])
                )
            else:
                raise ValueError(
                    "Unsupported sampling method. "
                    "Use 'midpoint' or 'random'."
                )
            stratified_samples.append(stratum_samples)

        meshgrid = np.meshgrid(*stratified_samples)
        grid_points = np.vstack(
            [mg.flatten() for mg in meshgrid]
        ).T
        selected_indices = np.random.choice(
            grid_points.shape[0], n, replace=True)

        xs = grid_points[selected_indices]

        return xs, f(xs)
