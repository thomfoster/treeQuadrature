import vegas

from .base_class import Integrator
from ..example_problems import Problem
from ..utils import ResultDict


class ShapeAdapter:
    def __init__(self, f):
        self.f = f

    def __call__(self, X):
        return self.f(X)[0, 0]


class VegasIntegrator(Integrator):
    """
    Integrator that uses the VEGAS algorithm for adaptive Monte Carlo integration.

    Parameters
    ----------
    N : int
        Number of samples to draw per iteration.
    n_iter : int
        Number of adaptive iterations to perform.
    """

    def __init__(self, N: int, n_iter: int, n_adaptive: int = 5):
        self.N = N
        self.n_iter = n_iter
        self.n_adaptive = n_adaptive

    def __call__(self, problem: Problem, return_N: bool = False) -> ResultDict:
        """
        Perform the integration process using the VEGAS algorithm.

        Parameters
        ----------
        problem : Problem
            The integration problem
        return_N : bool, optional
            If True, return the number of samples used.

        Return
        -------
        dict
            with the following keys:
            - 'estimate' (float) : estimated integral value
            - 'n_evals' (int) :  number of function estiamtions, if return_N is True
        """
        domain_bounds = []
        for i in range(problem.D):
            domain_bounds.append([problem.lows[i], problem.highs[i]])

        integ = vegas.Integrator(domain_bounds)
        # adjust the shape of output to suit vegas.Integrator
        f = ShapeAdapter(problem.integrand)

        # adaptive runs
        integ(f, nitn=self.n_adaptive, neval=self.N)

        result_vegas = integ(f, nitn=self.n_iter, neval=self.N)
        estimate = result_vegas.mean

        ret = ResultDict(estimate)
        if return_N:
            ret["n_evals"] = self.N * (self.n_iter + self.n_adaptive)
        return ret
