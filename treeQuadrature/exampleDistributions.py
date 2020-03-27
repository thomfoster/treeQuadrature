import numpy as np

from scipy.stats import multivariate_normal

# TODO: These multimodal distributions I've created - I'm not sure they're exactly 
# what the say they are. Check with Ben whether the which_dist approach works.


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


class MixtureDistribution:
    """Base class for our multimodal distributions"""
    def __init__(self, dims):
        self.dims = dims
        # Inherited classes define how to choose mixture dists
        self.dists = []

    def rvs(self, n_samples):
        dists = list(range(len(self.dists)))
        which_dist = np.random.choice(dists, size=(n_samples,))
        X = np.array([self.dists[i].rvs(1).reshape(-1,) for i in which_dist])
        return X
    
    def pdf(self, x):
        n_dists = len(self.dists)
        fvals = [dist.pdf(x).reshape(-1,) / n_dists for dist in self.dists]
        fvals = np.stack(fvals)
        sums = np.sum(fvals, axis=0)
        return sums.reshape(-1, 1)


class Camel(MixtureDistribution):
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


class QuadCamel(MixtureDistribution):
    '''
    Complementary test to the camel, more humps, more spaced out.

    It is 4 d-dimensiona gaussians, with sigma=1/(10sqrt(2)), 
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