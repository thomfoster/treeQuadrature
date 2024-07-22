from ..exampleProblems import Problem
from .integrator import Integrator
from ..samplers import Sampler, UniformSampler
from ..container import Container
from ..gaussianProcess import fit_GP, rbf_Integration

from sklearn.gaussian_process.kernels import RBF, Kernel
import numpy as np

default_sampler = UniformSampler()
default_kernel = kernel = RBF(1.0, (1e-2, 1e2))

class BayesMcIntegrator(Integrator):
    def __init__(self, N: int, kernel: Kernel=default_kernel, 
                 sampler: Sampler=default_sampler, 
                 n_tuning: int = 10, max_iter: int = 1000, factr: float = 1e7) -> None:
        """
        Initialise the BayesMcIntegrator.

        Parameters
        ----------
        N : int
            The number of evaluations.
        kernel : Kernel, optional
            The kernel to use for the Gaussian Process. 
            Default is RBF.
        sampler : Sampler, optional
            The sampler to use for generating samples. 
            Default is UniformSampler.
        base_N : int, optional
            The base number of samples to draw for the problem. 
            Default is 100.
        n_tuning : int, optional
            The number of tuning steps for the Gaussian Process. 
            Default is 10.
        max_iter : int, optional
            The maximum number of iterations for 
            fitting the Gaussian Process. 
            Default is 1000.
        factr : float, optional
            The factor for convergence criterion.
            Default is 1e7.
        """
        self.N = N
        self.kernel = kernel
        self.sampler = sampler
        self.n_tuning = n_tuning
        self.max_iter = max_iter
        self.factr = factr

    def __call__(self, problem: Problem, return_N: bool=False, return_std: bool=False) -> dict:
        if hasattr(problem, 'rvs'):
            X = problem.rvs(self.N)
        else:
            X = self.sampler.rvs(self.N, problem)

        y = problem.integrand(X)

        gp = fit_GP(X, y, kernel, self.n_tuning, self.max_iter, self.factr)
        
        # create empty container for convenience
        X_empty = np.empty(shape=(0, problem.D))
        y_empty = np.empty(shape=(0, 1))
        cont = Container(X_empty, y_empty, mins=problem.lows, maxs=problem.highs)

        result = {}
        if isinstance(self.kernel, RBF):
            integral_result = rbf_Integration(gp, cont, return_std)
            if return_std:
                result['estimate'] = integral_result[0]
                result['std'] = integral_result[1]
            else:
                result['estimate'] = integral_result
        else:
            raise NotImplementedError
        
        
        if return_N:
            result['n_evals'] = self.N

        return result