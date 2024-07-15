from treeQuadrature import exampleDistributions

# for Gaussian
import numpy as np
from scipy.stats import multivariate_normal

"""
Defines the specific problems we want results for.
"""


class Problem:
    '''
    base class for integration problems

    Attributes
    ----------
    D : int
        dimension of the problem
    d : Distribution or MixtureDistribution
        the likelihood function in integral
    lows, highs : float
        the lower and upper bound of integration domain
        assumed to be the same for each dimension
    p : Distribution or MixtureDistribution
        the density function in integral
    answer : float
        the True solution to int p.pdf(x) * d.pdf(x) dx
        integrated over [lows, highs]

    Methods
    -------
    pdf(X)
        X : numpy array of shape (N, D)
            each row is a sample
        return : numpy array of shape (N, 1)
            the value of p.pdf(x) * d.pdf(x) at samples in X
    '''
    def __init__(self, D):
        self.D = D
        self.d = None
        self.lows = None
        self.highs = None
        self.p = None
        self.answer = None

    def pdf(self, X):
        # check dimensions of X
        flag = True
        if X.ndim == 1 and self.D != 1 and X.shape[0] != self.D:
            flag = False
        elif X.ndim == 2 and X.shape[1] != self.D:
            flag = False

        assert flag, 'the dimension of X should match the dimension of the problem'

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.p: # Combined pdf ie d(x) * p(x)
            return self.d.pdf(X) * self.p.pdf(X)
        else: # when p is not defined, simply use d
            return self.d.pdf(X)



class SimpleGaussian(Problem):
    """
    Likelihood: N(0, 1/(10*sqrt(2)))
    Prior: U([-1, 1])
    """

    def __init__(self, D):
        self.D = D
        self.d = exampleDistributions.MultivariateNormal(
            D=D, mean=[0.0] * D, cov=1 / 200)
        self.lows = [-1.0] * D
        self.highs = [1.0] * D
        self.p = exampleDistributions.Uniform(
            D=D, low=self.lows, high=self.highs)

        # Truth
        self.answer = 1 / (2.0**D)

class Gaussian(Problem):
    """
    Integration of general Gaussian pdf on rectangular bounds

    Arguments
    ---------
    D : int
        Dimension of the problem
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
        Arguments
        ---------
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
        mu = self._handle_bound(mu, D, 0)
        Sigma = self._handle_Sigma(Sigma, D)

        self.lows = self._handle_bound(lows, D, -np.inf)
        self.highs = self._handle_bound(highs, D, np.inf)
        self.D = D
        self.d = exampleDistributions.MultivariateNormal(
            D=D, mean=mu, cov=Sigma)
        self.p = None

        self.answer = self._integrate()

    @staticmethod
    def _handle_bound(value, D, default_value):
        if value is None:
            return [default_value] * D
        elif isinstance(value, (int, float)):
            return [value] * D
        elif isinstance(value, (list, np.ndarray)) and len(value) == D:
            return np.array(value)
        else:
            raise ValueError(
                "value must be a number, list, or numpy.ndarray"
                f"with length {D} when given as a list or numpy.ndarray"
            )
        
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

    def _integrate(self):
        """
        Calculate the integral of the Gaussian pdf over the hyper-rectangular bounds defined by lows and highs.
        
        Returns
        -------
        float
            The integral of the Gaussian pdf over the specified bounds.
        """
        # fetch to multivariate Gaussian object
        rv = self.d.d
        
        # Calculate the CDF values at the bounds
        lower_cdf = rv.cdf(self.lows)
        upper_cdf = rv.cdf(self.highs)
        
        # The integral over the hyper-rectangle is the difference of the CDFs
        integral_value = upper_cdf - lower_cdf
        
        return integral_value

class Camel(Problem):
    """
    Likelihood: Two Gaussians 1/3 and 2/3 along unit diagonal. cov = 1/200.
    Prior: U([0, 1])
    """

    def __init__(self, D):
        self.D = D
        self.d = exampleDistributions.Camel(D)
        self.lows = [-0.5] * D
        self.highs = [1.5] * D
        self.p = exampleDistributions.Uniform(
            D=D, low=self.lows, high=self.highs)

        # Truth
        self.answer = 1 / (2.0**D)


class QuadCamel(Problem):
    """
    A challenging problem with more modes, more spread out, than those in
    Camel.

    Likelihood: 4 Gaussians 2,4,6,8 units along diagonal. cov = 1/200.
    Prior: U([0, 10])
    """

    def __init__(self, D):
        self.D = D
        self.d = exampleDistributions.QuadCamel(D)
        self.lows = [0.0] * D
        self.highs = [10.0] * D
        self.p = exampleDistributions.Uniform(
            D=D, low=self.lows, high=self.highs)

        # Truth
        self.answer = 1 / (10.0**D)
