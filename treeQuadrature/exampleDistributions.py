import numpy as np

from scipy.stats import multivariate_normal

class Uniform:
    def __init__(self, D, low, high):
        self.D = D
        self.low = low
        self.high = high

    def rvs(self, n):
        return np.random.uniform(low=self.low, high=self.high, size=(n, self.D))

    def pdf(self, X):
        if np.all(X > self.low) and np.all(X < self.high):
            pdfVal = (self.high - self.low)**self.D
            return np.array([1/pdfVal for x in X]).reshape(X.shape[0], 1)
        else:
            return np.zeros(shape=(X.shape[0], 1))

class MultivariateNormal:
    '''Example of how to wrap function to use in this module.'''
    def __init__(self, D, mean, cov):
        self.D = D
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


class RandomMultimodal:
    '''Unnormalised multimodal gaussian distro'''

    def __init__(self, dims, n_modes):
        self.dims = dims
        self.n_modes = n_modes

        # choose modes somewhere between {-10,10}^dims
        means = np.random.uniform(size=(n_modes, dims))
        means = 20*means - 10
        self.dists = [multivariate_normal(mean=mean) for mean in means]

    def rvs(self, n_samples):
        n_samples_per_mode = n_samples // self.n_modes
        samples = np.vstack(
            [dist.rvs(n_samples_per_mode).reshape(-1, self.dims) for dist in self.dists])
        return samples

    def pdf(self, x):
        fvals = [dist.pdf(x).reshape(-1,) for dist in self.dists]
        fvals = np.stack(fvals)
        sums = np.sum(fvals, axis=0)
        return sums


class Camel:
    '''
    Specific multimodal distro used as a test in the literature.

    It is a pair of d dimensional gaussians, with sigma=1/10sqrt(2), 
    placed at 1/3 and 2/3 along the unit hypercube diagonal.
    '''

    def __init__(self, dims):
        self.dims = dims

        mean1 = (1/np.sqrt(dims)) * (1/3)
        mean2 = (1/np.sqrt(dims)) * (2/3)

        cov = 1/200

        self.dists = [
            multivariate_normal(mean=[mean1]*dims, cov=cov),
            multivariate_normal(mean=[mean2]*dims, cov=cov)
            ]

    def rvs(self, n_samples):
        n_samples_per_mode = n_samples // 2
        samples = np.vstack(
            [dist.rvs(n_samples_per_mode).reshape(-1, self.dims) for dist in self.dists]
        )
        return samples

    def pdf(self, x):
        fvals = [dist.pdf(x).reshape(-1,) for dist in self.dists]
        fvals = np.stack(fvals)
        sums = np.sum(fvals, axis=0)
        return sums


class QuadCamel:
    '''
    Complementary test to the camel, more humps, more spaced out.

    It is 4 d dimensiona gaussians, with sigma=1/(10sqrt(2)), 
    placed at 1,3,5,7 along the unit hypercube.
    '''

    def __init__(self, dims):
        self.dims = dims

        mean1 = (1/np.sqrt(dims)) * 1
        mean2 = (1/np.sqrt(dims)) * 3
        mean3 = (1/np.sqrt(dims)) * 5
        mean4 = (1/np.sqrt(dims)) * 7

        cov = 1/200

        self.dists = [
            multivariate_normal(mean=[mean1]*dims, cov=cov),
            multivariate_normal(mean=[mean2]*dims, cov=cov),
            multivariate_normal(mean=[mean3]*dims, cov=cov),
            multivariate_normal(mean=[mean4]*dims, cov=cov)
            ]

    def rvs(self, n_samples):
        n_samples_per_mode = n_samples // 4
        samples = np.vstack(
            [dist.rvs(n_samples_per_mode).reshape(-1, self.dims) for dist in self.dists]
        )
        return samples

    def pdf(self, x):
        fvals = [dist.pdf(x).reshape(-1,) for dist in self.dists]
        fvals = np.stack(fvals)
        sums = np.sum(fvals, axis=0)
        return sums