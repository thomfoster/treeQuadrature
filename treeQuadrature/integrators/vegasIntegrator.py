import vegas

from .integrator import Integrator

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
    NITN : int
        Number of adaptive iterations to perform.
    """

    def __init__(self, N: int, NITN: int):
        self.N = N
        self.NITN = NITN

    def __call__(self, problem, return_N=False, return_all=False):
        """
        Perform the integration process using the VEGAS algorithm.

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
            The computed integral and optionally the number of samples.
        """
        integ = vegas.Integrator([[-1.0, 1.0]] * problem.D)
        f = ShapeAdapter(problem.pdf)
        G = integ(f, nitn=self.NITN, neval=self.N).mean

        ret = (G, self.N * self.NITN) if return_N or return_all else G
        return ret
