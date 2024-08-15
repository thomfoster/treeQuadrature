from abc import ABC, abstractmethod
import numpy as np

class Sampler(ABC):
    @abstractmethod
    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            *args, **kwargs) -> np.ndarray:
        """
        A method to generate random samples 

        Argument
        --------
        n : int 
            number of samples
        problem: Problem
            the integration problem being solved
        *args, **kwargs
            other necessary arguments and keyward arguments

        Return
        ------
        np.ndarray of shape (n, self.D)
            samples from the distribution
        """
        pass

    @staticmethod
    def handle_mins_maxs(mins : np.ndarray, maxs : np.ndarray):
        """
        Check the shapes of mins, maxs 

        Return
        ------
        tuple
        mins, maxs : np.ndarray
            reshaped to (D,) when necessary
        D : int
            the dimension of the sampler
        """
        if mins.shape != maxs.shape:
            raise ValueError('mins and maxs must have the same shape'
                             f'got mins {mins.shape}, maxs {maxs.shape}')
        
        if mins.ndim == 2 and mins.shape[1] == 1:
            mins = mins.reshape(-1)
            maxs = maxs.reshape(-1)
        elif mins.ndim != 1: 
            raise ValueError('mins and maxs must be one dimensional arrays,'
                             'got ')

        return mins, maxs, mins.shape[0]