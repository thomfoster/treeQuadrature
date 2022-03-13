import numpy as np


class SmcIntegrator:
    def __init__(self, N):
        """
        Simplest possible integrator. Draw N samples from prior,
        take sample mean of liklihood values at these samples.
        """
        self.N = N

    def __call__(self, problem, return_N=False, return_all=False):
        xs = problem.p.rvs(self.N)
        ys = problem.d.pdf(xs).reshape(-1)
        G = np.mean(ys)

        ret = (G, self.N) if return_N else G
        ret = (G, self.N) if return_all else ret
        return ret
