import numpy as np
from scipy.stats import qmc

from .base_class import Problem
from . import distributions as dists
from ..utils import handle_bound



class BayesProblem(Problem):
    '''
    base class for integration problems, 
    with a prior density p and a likelihood d

    Attributes
    ----------
    D : int
        dimension of the problem
    d : Distribution
        the likelihood function in integral
    lows, highs : float or list or np.array
        the lower and upper bound of integration domain
        must have length D
    p : Distribution, optional
        the density function in integral
    answer : float
            the True solution to int p.pdf(x) * d.pdf(x) dx
            integrated over [lows, highs]

    Methods
    -------
    pdf(X)
        input : numpy.ndarray of shape (N, D)
            each row is a sample
        return : numpy.ndarray of shape (N, 1)
            the value of p.pdf(x) * d.pdf(x) at samples in X
    '''
    def __init__(self, D: int,
                 d: dists.Distribution, 
                 lows, highs):
        """
        Parameters
        ----------
        D : int
            dimension of the problem
        d : Distribution
            the likelihood function in integral
        lows, highs : float or list or np.array
            the lower and upper bound of integration domain
            must have length D
        """
        super().__init__(D, lows, highs)
        self.d = d
        self.p = None

    def integrand(self, X):
        # check dimensions of X
        xs = self.handle_input(X)

        if self.p: # Combined pdf ie d(x) * p(x)
            return self.d.pdf(xs) * self.p.pdf(xs)
        else: # when p is not defined, simply use d
            return self.d.pdf(xs)

    def rvs(self, n):
        """
        Generate random samples from the likelihood distribution

        Argument
        --------
        n : int 
            number of samples

        Return
        ------
        np.ndarray of shape (n, self.D)
            samples from the distribution
        """
        return self.d.rvs(n)





class SimpleGaussian(BayesProblem):
    """
    Attributes
    ----------
    d : Distribution
        Likelihood, N(0, 1/(10*sqrt(2)))
    p : Distribution
        prior, U([-1, 1]^D)
    D : int
        dimension of the problem
    lows, highs : numpy.ndarray
        the lower and upper bound of integration domain,
        from -1.0 to 1.0 in each dimension
    answer : float
        1 / (2.0**self.D)
    """

    def __init__(self, D):
        """
        Parameter
        ----------
        D : int
            dimension of the problem
        """
        super().__init__(D, d=dists.MultivariateNormal(
            D=D, mean=[0.0] * D, cov=1 / 200), 
            lows = -1.0, highs= 1.0)
        
        self.p = dists.Uniform(
            D=D, low=self.lows, high=self.highs)
        # Truth
        self.answer = 1 / (2.0**D)

    def __str__(self) -> str:
        return f'SimpleGaussian(D={self.D})'

class Gaussian(BayesProblem):
    """
    Integration of general Gaussian pdf on rectangular bounds

    Arguments
    ---------
    D : int
        Dimension of the problem
    d : Distribution
        Multivariate Gaussian with 
          mean mu and covariance Sigma
    mu : numpy.ndarray
        Mean vector
    Sigma : numpy.ndarray
        Covariance matrix
    lows, highs : numpy.ndarray
        Bounds of the integration domain
    answer : float
        The true solution
    """

    def __init__(self, D, mu=None, Sigma=None, lows=None, highs=None):
        """
        Parameters
        ----------
        D : int
            Dimension of the problem
        mu, lows, highs : number or list or numpy.ndarray, optional
            if a number given, used for each dimension
            if list or array given, must have length D
            mu defaults to 0
            lows and highs defaults to +- np.inf 
        Sigma : number of numpy.ndarray
            if a number given, covariance set to Sigma * I
            if array given, must have shape (D, D)
            Sigma defaults to I
        """
        # Value checks
        self.mu = handle_bound(mu, D, 0)
        self.Sigma = Gaussian._handle_Sigma(Sigma, D)

        super().__init__(D, d = dists.MultivariateNormal(
            D=D, mean=self.mu, cov=self.Sigma), 
            lows=lows, highs=highs)

        self.answer = self._integrate(self.lows, self.highs)
        
    @staticmethod
    def _handle_Sigma(value, D):
        if value is None:
            return np.eye(D)
        elif isinstance(value, (int, float)):
            return value * np.eye(D)
        elif isinstance(value, np.ndarray) and value.shape == (D, D):
            return value
        else:
            raise ValueError(
                "value must be a number, or numpy.ndarray"
                f"with shape ({D}, {D}) when given as a list or numpy.ndarray"
            )

    def _integrate(self, lows, highs, num_samples=8192):
        """
        Calculate the integral of the Gaussian pdf over the 
        hyper-rectangular bounds defined by lows and highs.
        
        Returns
        -------
        float
            The integral of the Gaussian pdf over the specified bounds.
        """
        # fetch to multivariate Gaussian object
        rv = self.d.d
        
        dim = len(lows)
        
        # Generate QMC samples within the unit hypercube
        sampler = qmc.Sobol(d=dim, scramble=True)
        unit_samples = sampler.random(num_samples)
        
        samples = qmc.scale(unit_samples, lows, highs)
        
        pdf_values = rv.pdf(samples)
        volume = np.prod(highs - lows)
        integral_value = np.mean(pdf_values) * volume
        
        return integral_value
    
    def __str__(self) -> str:
        return f'Gaussian(D={self.D}, mean={self.mu}, cov={self.Sigma})'

class Camel(BayesProblem):
    """
    Attributes
    ----------
    d : Distribution
        Likelihood: mixture of Two Gaussians 
          centred at 1/3 and 2/3 along unit diagonal. cov = 1/200.
    p : Distribution
        Prior: U([-0.5, 1.5]^D)
    D : int
        dimension of the problem
    lows, highs : numpy.ndarray
        the lower and upper bound of integration domain,
        from -0.5 to 1.5 in each dimension
    answer : float
        1 / (2.0**self.D)
    """

    def __init__(self, D):
        """
        Parameter
        ---------
        D : int
            dimension of the problem
        """
        super().__init__(D, dists.Camel(D), 
                         lows=-0.5, highs=1.5)
        
        self.p = dists.Uniform(
            D=D, low=self.lows, high=self.highs)
        self.answer = 1 / (2.0**D)

    def __str__(self) -> str:
        return f'Camel(D={self.D})'


class QuadCamel(BayesProblem):
    """
    A challenging problem with more modes, more spread out, than those in
    Camel.

    Attributes
    ----------
    d : Distribution
        Likelihood: mixture of 4 Gaussians 
          2,4,6,8 units along diagonal. cov = 1/200.
    p : Distribution
        Prior: U([0, 10]^D)
    D : int
        dimension of the problem
    lows, highs : numpy.ndarray
        the lower and upper bound of integration domain,
        from 0 to 10 in each dimension
    answer : float
        1 / (10.0**D)
    """

    def __init__(self, D):
        super().__init__(D, d = dists.QuadCamel(D),
                         lows = 0.0, highs=10.0)
        
        self.p = dists.Uniform(
            D=D, low=self.lows, high=self.highs)
        self.answer = 1 / (10.0**D)

    def __str__(self) -> str:
        return f'QuadCamel(D={self.D})'