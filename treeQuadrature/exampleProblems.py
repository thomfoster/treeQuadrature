from treeQuadrature import exampleDistributions

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
    low, high : float
        the lower and upper bound of integration domain
        assumed to be the same for each dimension
    p : Distribution or MixtureDistribution
        the density function in integral
    answer : float
        the True solution to int p.pdf(x) * d.pdf(x) dx
        integrated over [low, high]

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
        self.low = None
        self.high = None
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
        self.low = -1.0
        self.high = 1.0
        self.p = exampleDistributions.Uniform(
            D=D, low=self.low, high=self.high)

        # Truth
        self.answer = 1 / (2.0**D)


class Camel(Problem):
    """
    Likelihood: Two Gaussians 1/3 and 2/3 along unit diagonal. cov = 1/200.
    Prior: U([0, 1])
    """

    def __init__(self, D):
        self.D = D
        self.d = exampleDistributions.Camel(D)
        self.low = -0.5
        self.high = 1.5
        self.p = exampleDistributions.Uniform(
            D=D, low=self.low, high=self.high)

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
        self.low = 0.0
        self.high = 10.0
        self.p = exampleDistributions.Uniform(
            D=D, low=self.low, high=self.high)

        # Truth
        self.answer = 1 / (10.0**D)
