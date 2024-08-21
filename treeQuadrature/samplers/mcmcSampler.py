from typing import Tuple
import numpy as np
from emcee import EnsembleSampler
from .sampler import Sampler

class McmcSampler(Sampler):
    """
    MCMC sampler that generates samples from 
    the modulus of f using the `emcee` package.
    """

    def __init__(self, n_walkers: int = 10, burning: int = 0):
        """
        Arguments
        ---------
        n_walkers : int, Optional
            Number of walkers in the MCMC sampling.
            Default is 10.
        burning : int, Optional
            Number of initial samples to discard.
            Defaults to 0.
        """
        self.n_walkers = n_walkers
        self.burning = burning

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray,
            f: callable, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate exactly n MCMC samples using the EnsembleSampler.

        Parameters
        ----------
        n : int 
            Number of samples to generate.
        mins, maxs : np.ndarray
            1-dimensional arrays of the lower and upper bounds.
        f : callable
            The integrand function from which to sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The generated samples and their corresponding function values.
        """
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        def log_prob(x):
            # Calculate the log probability, using abs(f) for sampling
            within_bounds = np.all((x >= mins) & (x <= maxs))
            if not within_bounds:
                return -np.inf  # log(0) = -inf

            f_val = np.abs(f(x.reshape(1, -1)))
            
            # Avoid log of zero or negative values
            if f_val[0] <= 0:
                return -np.inf
            
            return np.log(f_val[0])

        # Initialize walkers
        p0 = np.random.uniform(mins, maxs, size=(self.n_walkers, D))

        # Calculate how many steps we need to take to get exactly n samples
        nsteps = (n // self.n_walkers) + self.burning

        # Initialize and run the sampler
        sampler = EnsembleSampler(self.n_walkers, D, log_prob)
        sampler.run_mcmc(p0, nsteps, progress=False)

        # Extract exactly n samples after burn-in
        flat_samples = sampler.get_chain(discard=self.burning, flat=True)
        samples = flat_samples[:n]

        # Evaluate the function on the final samples
        final_values = np.abs(f(samples))

        return samples, final_values