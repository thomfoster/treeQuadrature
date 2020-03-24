import numpy as np

from treeQuadrature.exampleDistributions import MultivariateNormal, Uniform

class SimpleGaussian:
    '''
    Likelihood: N(0, 1/(10*sqrt(2)))
    Prior: U(-1, 1)
    '''
    def __init__(self, D):
        self.D = D
        self.d = MultivariateNormal(D=D, mean=[0.0]*D, cov=1/200)
        self.low = -1.0
        self.high = 1.0
        self.p = Uniform(D=D, low=self.low, high=self.high)
        
        # Calculate truth
        self.answer = 1/(2.0**D)
        
    def pdf(self, X):
        # Combined pdf ie d(x)*p(x)
        return self.d.pdf(X) * self.p.pdf(X)