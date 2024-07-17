import numpy as np

from abc import ABC, abstractmethod
from ..container import Container
from typing import Any, Callable

# abstract class for any integrator
class ContainerIntegral(ABC):
    @abstractmethod
    def containerIntegral(self, container: Container, f: Callable[..., np.ndarray], 
                          **kwargs: Any) -> float:
        """
        Arguments
        ---------
        container: Container
            the container on which the integral of f should be evaluated
        f : function
            takes X : np.ndarray and return np.ndarray, 
            see pdf method of Distribution class in exampleDistributions.py
        kwargs : Any
            other arguments allowed, must set default values
        
        Return
        ------
        float
            value of the integral of f on the container
        """
        pass