import numpy as np

class SmcIntegrator:
    def __init__(self, N):
        """
        Simplest possible integrator. Draw N samples from prior, 
        take sample mean of liklihood values at these samples.
        """
        self.N = N

    def __call__(self, problem):
        xs = problem.p.rvs(self.N)
        ys = problem.d.pdf(xs).reshape(-1)
        return np.mean(ys)