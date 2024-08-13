from abc import ABC, abstractmethod
import numpy as np

from ..exampleProblems import Problem

class Sampler(ABC):
    @abstractmethod
    def rvs(self, n: int, problem: Problem,
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