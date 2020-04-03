import vegas
import numpy as np

class ShapeAdapter:
    def __init__(self, f):
        self.f = f
    def __call__(self, X):
        return self.f(X)[0,0]

class VegasIntegrator:
    def __init__(self, N, NITN):
        """
        Runs the vegas algorithm on the problem.
        
        args:
        -------
        N: Int, Number of samples to draw per iteration.
        NITN: Int, Number of adaptive iterations to perform.
        """
        self.N = N
        self.NITN = NITN

    def __call__(self, problem, return_N=False, return_all=False):
        integ = vegas.Integrator([[-1.0, 1.0]]*problem.D)
        f = ShapeAdapter(problem.pdf)
        G = integ(f, nitn=self.NITN, neval=self.N).mean

        ret = (G, self.N * self.NITN) if return_N else G
        ret = (G, self.N * self.NITN) if return_all else ret
        return ret