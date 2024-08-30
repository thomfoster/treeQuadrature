from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List

class Sampler(ABC):
    @abstractmethod
    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray, 
            f: callable, 
            *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        A method to generate random samples 

        Argument
        --------
        n : int 
            number of samples
        mins, maxs : np.ndarray
            1 dimensional arrays of the lower bounds
            and upper bounds
        f : function
            the integrand
        *args, **kwargs
            other necessary arguments and keyward arguments

        Return
        ------
        tuple 
            (xs, ys)
            xs is np.ndarray of shape (n, D)
            ys is np.ndarray of shape (n, )
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
    
    @staticmethod
    def subdivide_domain(strata_per_dim: int, mins: np.ndarray, 
                         maxs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Subdivide the domain into smaller strata. \n
        For stratified samplers. 

        Parameters
        ----------
        strata_per_dim : int
            Number of strata per dimension.
        mins : np.ndarray
            Lower bounds of the domain.
        maxs : np.ndarray
            Upper bounds of the domain.

        Returns
        -------
        list of tuples
            Subdivided strata as a list of (low, high) tuples.
        """
        strata = []
        for indices in np.ndindex(*(strata_per_dim,) * len(mins)):
            sub_low = mins + (maxs - mins) * np.array(indices) / strata_per_dim
            sub_high = mins + (maxs - mins) * (np.array(indices) + 1) / strata_per_dim
            strata.append((sub_low, sub_high))
        return strata