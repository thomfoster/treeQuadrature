from abc import ABC, abstractmethod
import numpy as np

class Sampler(ABC):
    @abstractmethod
    def rvs(self, n: int, *args, **kwargs) -> np.ndarray:
        """
        A method to generate random samples 

        Argument
        --------
        n : int 
            number of samples
        *args, **kwargs
            other necessary arguments and keyward arguments

        Return
        ------
        np.ndarray of shape (n, self.D)
            samples from the distribution
        """
        pass