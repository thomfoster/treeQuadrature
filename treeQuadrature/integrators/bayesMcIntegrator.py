from ..exampleProblems import Problem
from .integrator import Integrator
from ..samplers import Sampler, UniformSampler
from ..container import Container
from ..gaussianProcess import rbf_Integration, IterativeGPFitting, SklearnGPFit

from sklearn.gaussian_process.kernels import RBF, Kernel
import numpy as np

default_sampler = UniformSampler()
default_kernel = kernel = RBF(1.0, (1e-2, 1e2))

class BayesMcIntegrator(Integrator):
    def __init__(self, N: int, kernel: Kernel=default_kernel, 
                 sampler: Sampler=default_sampler, 
                 n_tuning: int = 10, max_iter: int = 1000, factr: float = 1e7, 
                 length: float=1.0, range: float=1e2) -> None:
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
        base_N : int, optional (Default=100)
            The base number of samples to draw for the problem. 
        n_tuning : int, optional (Default=10)
            The number of tuning steps for the Gaussian Process. 
        max_iter : int, optional (Default=1000)
            The maximum number of iterations for 
            fitting the Gaussian Process. 
        factr : float, optional (Default=1e7)
            The factor for convergence criterion.
        length: float, optional (Default=1.0)
            initial length scale of RBF kernel
        range : float, optional (Default=1e2)

        """
        self.N = N
        self.kernel = kernel
        self.sampler = sampler
        self.n_tuning = n_tuning
        self.max_iter = max_iter
        self.factr = factr
        self.length = length
        self.range = range

    def __call__(self, problem: Problem, return_N: bool=False, return_std: bool=False) -> dict:
        # create a container with samples
        if hasattr(problem, 'rvs'):
            X = problem.rvs(self.N)
        else:
            X = self.sampler.rvs(self.N, problem)

        y = problem.integrand(X)
        
        cont = Container(X, y, mins=problem.lows, maxs=problem.highs)

        gp_fitter = SklearnGPFit(n_tuning=self.n_tuning, max_iter=self.max_iter, 
                                 factr=self.factr)
        # draw all samples at once, so threshold does not matter
        iGp = IterativeGPFitting(n_samples=self.N, max_redraw=1, gp=gp_fitter, 
                                 performance_threshold=0.0, threshold_direction='up')
        iGp.fit(problem.integrand, cont, self.kernel)
        gp = iGp.gp

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