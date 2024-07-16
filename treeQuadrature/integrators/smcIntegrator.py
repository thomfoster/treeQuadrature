import numpy as np


class SmcIntegrator:
    """
    Simple integrator: Draw N samples from prior,
    take sample mean of likelihood values at these samples.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    """
    def __init__(self, N):
        self.N = N

    def __call__(self, problem, return_N=False, return_all=False):
        """
        Perform the integration process.

        Parameters
        ----------
        problem : Problem
            The integration problem
        return_N : bool, optional
            If True, return the number of samples used.
        return_all : bool, optional
            If True, return containers and their contributions to the integral

        Returns
        -------
        result : float or tuple
            The computed integral and optionally the number of samples
            and other details
        """
        # Draw N samples from the prior distribution
        xs = problem.p.rvs(self.N)
        # Evaluate the likelihood at these samples
        ys = problem.d.pdf(xs).reshape(-1)
        G = np.mean(ys)

        ret = (G, self.N) if return_N or return_all else G
        return ret
