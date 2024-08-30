from abc import ABC, abstractmethod
import numpy as np

from ..utils import handle_bound


def ensure_2d_output(func):
    """
    Decorator to ensure that the output of the integrand 
    function is a 2D array of shape (N, 1).
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        elif result.ndim != 2 or result.shape[1] != 1:
            raise ValueError(f"The integrand must return a 2D array of shape (N, 1), "
                             f"but got an array with shape {result.shape}")
        return result
    return wrapper


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
    @ensure_2d_output
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
            2-dimensional array of shape (N, 1)
            each entry is f(x_i) where x_i is the ith row of X;
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
