from . import exampleDistributions as dists
from .utils import handle_bound
from .container import Container

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc

"""
Defines the specific problems we want results for.
"""

class Problem(ABC):
    '''
    base class for integration problems

    Attributes
    ----------
    D : int
        dimension of the problem
    lows, highs : np.ndarray
        the lower and upper bound of integration domain
        assumed to be the same for each dimension
    answer : float
        the True solution to int integrand(x) dx

    Methods
    -------
    integrand(X) 
        the function being integrated
        MUST be implemented by subclasses
        input : numpy.ndarray of shape (N, D)
            each row is a sample
        return : numpy.ndarray of shape (N, 1)
            the value of p.pdf(x) * d.pdf(x) at samples in X
    
    rvs(n)
        generate random samples in the integraiton domain 
        input : numpy.ndarray of shape (N, D)
            each row is a sample
        return : numpy.ndarray of shape (N, n)
            the value of p.pdf(x) * d.pdf(x) at samples in X
    '''
    def __init__(self, D, lows, highs):
        """
        Arguments
        ---------
        D : int
            dimension of the problem
        lows, highs : int or float or list or np.ndarray
            the lower and upper bound of integration domain
            assumed to be the same for each dimension
        """
        self.D = D
        self.lows = handle_bound(lows, D, -np.inf)
        self.highs = handle_bound(highs, D, np.inf)
        self.answer = None
        
    @abstractmethod
    def integrand(self, X, *args, **kwargs) -> np.ndarray:
        """
        must be defined, the function being integrated
        
        Argument
        --------
        X : numpy.ndarray
            each row is an input vector

        Return
        ------
        numpy.ndarray
            1-dimensional array of the same length as X
            each entry is f(x_i) where x_i is the ith row of X;
            or 2-dimensional array of shape (N, 1)
        """
        pass

    def __str__(self) -> str:
        return f'Problem(D={self.D}, lows={self.lows}, highs={self.highs})'
    
    def handle_input(self, xs) -> np.ndarray:
        """
        Check the shape of xs and 
        change xs to the correct shape (N, D)

        Parameter
        --------
        xs : numpy.ndarray
            the array to be handled

        Return
        numpy.ndarray
            the handled array
        """
        if isinstance(xs, list):
            xs = np.array(xs)
        elif not isinstance(xs, np.ndarray):
            raise TypeError('xs must be either a list or numpy.ndarray')
        
        if xs.ndim == 2 and xs.shape[1] == self.D:
            return xs
        elif xs.ndim == 1 and xs.shape[0] == self.D: # array with one sample
            return xs.reshape(1, -1)
        else:
            raise ValueError('xs must be either two dimensional array of shape (N, D)'
                             'or one dimensional array of shape (D,)'
                             f'got shape {xs.shape}')

class RippleProblem(Problem):
    def __init__(self, D, a=3):
        super().__init__(D, lows=-10., highs=10.)
        self.a = a 
        temp = (1/(1 + 16 * self.a**2))**(self.D/4) * np.cos(self.D*np.arccos(np.sqrt((1/(1 + 16 * self.a**2))))/2)
        self.answer = np.sqrt(2*np.pi)**self.D * (1 + temp)/2

    
    def integrand(self, X) -> np.ndarray:
        """
        Ripple integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        norms = np.linalg.norm(X, axis=1)
        f = np.exp(-norms**2/2)*np.cos(self.a*norms**2)**2
        return np.array(f).reshape(-1,1)
    
    
    def __str__(self) -> str:
        return f'Ripple(D={self.D})'


class OscillatoryProblem(Problem):
    def __init__(self, D, u=0, a=None):
        super().__init__(D, lows=0., highs=1.)
        if a == None:
            self.a = 5/np.linspace(1,D,D)
        else:
            self.a = a
        self.u = u
        self.answer = self.compute_answer(a=self.a, u=self.u)

    def compute_answer(self, a, u):
        if len(a) > 1:
            term1 = self.compute_answer(a[:-1], u - 1/4 + a[-1]/(2*np.pi))
            term2 = self.compute_answer(a[:-1], u - 1/4)
            return (term1 - term2)/a[-1]
        else:
            return (np.sin(2*np.pi*u + a[0]) - np.sin(2*np.pi*u))/a[0]


    def integrand(self, X) -> np.ndarray:
        """
        Oscillatory integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        dotprods = np.array([np.dot(x,self.a) for x in X])
        f = np.cos(2*np.pi*self.u + dotprods)
        return np.array(f).reshape(-1,1)
    
    
    def __str__(self) -> str:
        return f'Oscillatory(D={self.D})'



class ProductPeakProblem(Problem):
    def __init__(self, D, u=None, a=None):
        super().__init__(D, lows=0., highs=1.)
        if a == None:
            self.a = np.array([1.]*D)
        else:
            self.a = a
        
        if u == None:
            self.u = np.linspace(0.2,0.8,D)
        else:
            self.u = u

        self.answer = np.prod(self.a * (np.arctan(self.a*(1-self.u)) + np.arctan(self.a*self.u)))


    def integrand(self, X) -> np.ndarray:
        """
        Product Peak integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        f = [1/np.prod(self.a**(-2) + (x - self.u)**2) for x in X]
        return np.array(f).reshape(-1,1)
    
    
    def __str__(self) -> str:
        return f'ProductPeak(D={self.D})'


class CornerPeakProblem(Problem):
    def __init__(self, D, a=None):
        super().__init__(D, lows=0., highs=1.)
        if a == None:
            self.a = np.array([1.]*D)
        else:
            self.a = a
        
        self.answer = self.compute_answer(a0 = 1, a = self.a)

    def compute_answer(self, a0, a):
        if len(a) == 1:
            return 1/(a0*(a0 + a[0]))
        else:
            term1 = self.compute_answer(a0 = a0, a = a[:-1])
            term2 = self.compute_answer(a0 = a0 + a[-1], a = a[:-1])
            return (term1 - term2)/(a[-1]*len(a))


    def integrand(self, X) -> np.ndarray:
        """
        Corner Peak integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        dotprods = np.array([np.dot(x,self.a) for x in X])
        f =(1 + dotprods)**(-self.D-1)
        return np.array(f).reshape(-1,1)
    
    
    def __str__(self) -> str:
        return f'CornerPeak(D={self.D})'


class C0Problem(Problem):
    def __init__(self, D, u=None, a=None):
        super().__init__(D, lows=0., highs=1.)
        if a == None:
            self.a = np.array([1.]*D)
        else:
            self.a = a
        
        if u == None:
            self.u = np.linspace(0.2,0.8,D)
        else:
            self.u = u

        self.answer = np.prod((2 - np.exp(-self.a * self.u) - np.exp(-self.a * (1 - self.u)))/self.a)


    def integrand(self, X) -> np.ndarray:
        """
        C0 integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        f = [np.exp(-np.sum(self.a * np.abs(x - self.u))) for x in X]
        return np.array(f).reshape(-1,1)
    
    
    def __str__(self) -> str:
        return f'C0function(D={self.D})'


class DiscontinuousProblem(Problem):
    def __init__(self, D, a=None):
        super().__init__(D, lows=0., highs=1.)

        if a == None:
            self.a = np.array([1.]*D)
        else:
            self.a = a

        self.u1 = 0.3
        self.u2 = 0.5

        if D == 1:
            self.answer = (np.exp(self.a[0] * self.u1) - 1)/self.a[0]
        else:
            term1 = (np.exp(self.a[0] * self.u1) - 1)/self.a[0]
            term2 = (np.exp(self.a[0] * self.u2) - 1)/self.a[1]
            term3 = np.prod((np.exp(self.a[2:]) - 1)/self.a[2:])
            self.answer = term1 * term2 * term3

    def integrand(self, X) -> np.ndarray:
        """
        Discontinuous integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        X = self.handle_input(X)
        dotprods = np.array([np.dot(x,self.a) for x in X])
        if self.D == 1:
            f = np.array([np.where(x[0] > self.u1, 0, 1) for x in X]) * np.exp(dotprods)
        else:
            f = np.array([np.where((x[0] > self.u1 or x[1] > self.u2), 0, 1) for x in X]) * np.exp(dotprods)

        return f.reshape(-1,1)
    
    def __str__(self) -> str:
        return f'Discontinuous(D={self.D})'

    def handle_input(self, xs) -> np.ndarray:
        """
        Check the shape of xs and 
        change xs to the correct shape (N, D)

        Parameter
        --------
        xs : numpy.ndarray
            the array to be handled

        Return
        numpy.ndarray
            the handled array
        """
        if isinstance(xs, list):
            xs = np.array(xs)
        elif not isinstance(xs, np.ndarray):
            raise TypeError('xs must be either a list or numpy.ndarray')
        
        if xs.ndim == 2 and xs.shape[1] == self.D:
            return xs
        elif xs.ndim == 1 and xs.shape[0] == self.D: # array with one sample
            return xs.reshape(1, -1)
        else:
            raise ValueError('xs must be either two dimensional array of shape (N, D)'
                             'or one dimensional array of shape (D,)')


class PyramidProblem(Problem):
    def __init__(self, D):
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = (2 ** self.D) / (self.D + 1)

    def integrand(self, X) -> np.ndarray:
        """
        Pyramid integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray
            1-dimensional array of the same length as X.
        """

        ys = 1 - np.max(np.abs(X), axis=1)
        return ys.reshape(-1, 1)
    
    def __str__(self) -> str:
        return f'Pyramid(D={self.D})'


class QuadraticProblem(Problem):
    def __init__(self, D):
        """
        Quadratic function sum x_i^2 
        on [-1.0, 1.0]^D 

        Parameters 
        ----------
        D : int
            dimension 
        """
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = self.exact_integral(self.lows, self.highs)

    def integrand(self, X) -> np.ndarray:
        """
        Quadratic integrand function for the sum of squares.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray of shape (N, 1)
            the function value evaluated at X
        """
        xs = self.handle_input(X)
        ys = np.sum(xs**2, axis=1)
        return ys.reshape(-1, 1)

    def exact_integral(self, mins, maxs):
        """
        Calculate the exact integral from mins to maxs for the sum of squares polynomial.

        Parameters
        ----------
        mins : numpy.ndarray
            Lower bounds of the integration.

        maxs : numpy.ndarray
            Upper bounds of the integration.

        Returns
        -------
        float
            The value of the integral.
        """
        D = len(mins)
        integral_sum = 0
        
        for i in range(D):
            term = (maxs[i]**3 - mins[i]**3) / 3
            product = np.prod([maxs[j] - mins[j] for j in range(D) if j != i])
            integral_sum += term * product
        
        return integral_sum

    def __str__(self) -> str:
        return f'QuadraticProblem(D={self.D})'


class ExponentialProductProblem(Problem):
    def __init__(self, D):
        """
        Exponential product 
        on [-1.0, 1.0]^D 

        Parameters 
        ----------
        D : int
            dimension 
        """
        super().__init__(D, lows=-1.0, highs=1.0)
        self.answer = self.exact_integral(self.lows, self.highs)

    def integrand(self, X) -> np.ndarray:
        """
        Exponential product integrand function.

        Parameters
        ----------
        X : numpy.ndarray
            Each row is an input vector.

        Returns
        -------
        numpy.ndarray
            1-dimensional array of the same length as X.
        """
        xs = self.handle_input(X)
        ys = np.prod(np.exp(xs), axis=1)
        return ys.reshape(-1, 1)

    def exact_integral(self, mins, maxs):
        """
        Calculate the exact integral from mins to maxs for the product of exponentials.

        Parameters
        ----------
        mins : numpy.ndarray
            Lower bounds of the integration.

        maxs : numpy.ndarray
            Upper bounds of the integration.

        Returns
        -------
        float
            The value of the integral.
        """
        return np.prod([np.exp(maxs[i]) - np.exp(mins[i]) for i in range(len(mins))])

    def __str__(self) -> str:
        return f'ExponentialProductProblem(D={self.D})'


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
        
        # Number of dimensions
        dim = len(lows)
        
        # Generate QMC samples within the unit hypercube
        sampler = qmc.Sobol(d=dim, scramble=True)
        unit_samples = sampler.random(num_samples)
        
        # Scale samples to fit within the specified bounds
        samples = qmc.scale(unit_samples, lows, highs)
        
        # Evaluate the Gaussian PDF at the QMC sample points
        pdf_values = rv.pdf(samples)
        
        # Calculate the volume of the hyper-rectangle
        volume = np.prod(highs - lows)
        
        # Estimate the integral using the average of the PDF values scaled by the volume
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