import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import multivariate_normal

# TODO: These multimodal distributions I've created - I'm not sure they're
# exactly what the say they are. Check with Ben whether the which_dist approach
# works.

class Distribution(ABC):
    """
    An abstract framework for distributions
    """
    def __init__(self, D):
        self.D = D  # dimensions of the distribution

    @abstractmethod
    def rvs(self, *args, **kwargs) -> np.ndarray:
        """
        Generate random variates of the distribution.

        Args:
            *args, **kwargs: Additional arguments and keyword arguments for generating the random variates.

        Returns: numpy array
            contains samples from the distribution.
        """
        pass

    @abstractmethod
    def pdf(self, X, *args, **kwargs) -> np.ndarray:
        """
        Calculate the probability density function (pdf) at given point x.

        Args:
            X: array of points on which the pdf value should be evaluated
            *args, **kwargs: Additional arguments and keyword arguments for the pdf computation.

        Returns: numpy array
            The probability densities at x_i (rows of X)
        """
        pass

class Uniform(Distribution):
    """
    Uniform distribution

    Attributes
    ----------
    low, high : float or numpy array of shape (D, )
        the lower and uppper bounds
        if float given, the same bound used for all dimensions
    """
    def __init__(self, D, low, high):
        super().__init__(D=D)
        self.low = self._handle_low_high(low)
        self.high = self._handle_low_high(high)

    def _handle_low_high(self, value):
        if isinstance(value, (int, float)):
            return np.array([value] * self.D)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value)

    def rvs(self, n):
        return np.random.uniform(
            low=self.low, high=self.high, size=(
                n, self.D))

    def pdf(self, X):
        within_bounds = np.all((X >= self.low) & (X <= self.high), axis=1)
        pdfVal = np.prod(self.high - self.low)  # Volume of the hyperrectangle

        # Calculate the PDF values
        pdf_values = np.where(within_bounds, 1 / pdfVal, 0).reshape(X.shape[0], 1)
        return pdf_values


class MultivariateNormal(Distribution):
    '''
    Multivariate Normal/Gaussian distribution

    Attributes
    ----------
    mean : numpy array of shape (D, )
        the mean vector of Gaussian distribution
    cov : float, or numpy array of shape (D, D)
        the covariance matrix
        if float given, covariance matrix is cov * np.eye(D)
    '''

    def __init__(self, D, mean, cov):
        super().__init__(D=D)
        self.mean = mean
        assert np.array(mean).shape[0] == D
        self.cov = cov
        self.d = multivariate_normal(mean=mean, cov=cov)

    def rvs(self, n):
        '''Ensure resulting array is 2D'''
        return self.d.rvs(n).reshape(-1, self.D)

    def pdf(self, X):
        '''Ensure resulting array is 2D'''
        return self.d.pdf(X).reshape(-1, 1)


class MixtureDistribution:
    """
    Base class for our multimodal distributions

    Attribtues
    ----------
    D : int
        dimension of the problem
    dists : list
        component distributions
    weights : list 
        weights of each component distribution
    
    Methods
    -------
    rvs(n_samples)
        return n_samples random samples drawn from the mixture
    pdf(x)
        the probability density function of the mixture
    """

    def __init__(self, D, weights):
        self.D = D
        # Inherited classes define how to choose mixture dists
        self.dists = []
        self.weights= weights

    def rvs(self, n_samples):
        if n_samples == 0:
            return np.empty(shape=(0, self.D))

        dists = list(range(len(self.dists)))
        which_dist = np.random.choice(dists, size=(n_samples,), p=self.weights)
        X = np.array([self.dists[i].rvs(1).reshape(-1,) for i in which_dist])
        return X

    def pdf(self, x):
        fvals = [self.weights[i] * dist.pdf(x).reshape(-1,) for i, dist in enumerate(self.dists)]
        fvals = np.stack(fvals)
        sums = np.sum(fvals, axis=0)
        return sums.reshape(-1, 1)


class Camel(MixtureDistribution):
    '''
    Specific multimodal distribution used as a test in the literature.

    A pair of D-dimensional gaussian distributions, 
        with shared covariance sigma=1/200 I,
        and means placed at 1/3 and 2/3 along the unit hypercube diagonal.
    
    Default weights are equal
    '''

    def __init__(self, D, weights=None):
        self.D = D

        if weights is None:
            self.weights = [1/2] * 2
        else:
            assert len(weights) == 2, 'weights of Camel must have length 2'
            assert sum(weights) == 1, 'sum of weights must be 1'
            self.weights = weights

        mean1 = (1 / np.sqrt(D)) * (1 / 3)
        mean2 = (1 / np.sqrt(D)) * (2 / 3)

        cov = 1 / 200

        self.dists = [
            multivariate_normal(mean=[mean1] * D, cov=cov),
            multivariate_normal(mean=[mean2] * D, cov=cov)
        ]


class QuadCamel(MixtureDistribution):
    '''
    Complementary test to the camel, more humps, more spaced out.

    4 D-dimensiona gaussian distributions, 
        with shared covariance 1/200 I,
        and means placed at 1,3,5,7 along the unit hypercube.

    Default weights are equal
    '''

    def __init__(self, D, weights=None):
        self.D = D

        if weights is None:
            self.weights = [1/4] * 4
        else:
            assert len(weights) == 4, 'weights of QuadCamel must have length 4'
            assert sum(weights) == 1, 'sum of weights must be 1'
            self.weights = weights

        mean1 = (1 / np.sqrt(D)) * 2
        mean2 = (1 / np.sqrt(D)) * 4
        mean3 = (1 / np.sqrt(D)) * 6
        mean4 = (1 / np.sqrt(D)) * 8

        cov = 1 / 200

        self.dists = [
            multivariate_normal(mean=[mean1] * D, cov=cov),
            multivariate_normal(mean=[mean2] * D, cov=cov),
            multivariate_normal(mean=[mean3] * D, cov=cov),
            multivariate_normal(mean=[mean4] * D, cov=cov)
        ]
