import numpy as np

from abc import ABC, abstractmethod
from ..container import Container
from typing import Any, Callable

# If using TreeIntegrator and the containerIntegral 
# requires additional parameters other than container, f, return_std,
# please define a class fuction get_additional_params(self) that returns
# a dictionary of parameter names and values

class ContainerIntegral(ABC):
    """
    Abstract Base class for integrating a function on a container
    """
    @abstractmethod
    def containerIntegral(self, container: Container, f: Callable[..., np.ndarray], 
                          **kwargs: Any) -> dict:
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
        dict
            - integral (float) value of the integral of f on the container
            - and other necessary results
        """
        pass

    def __str__(self):
        if hasattr(self, 'name'):
            return f"ContainerIntegral--{self.name}"
        else:
            return 'ContainerIntegral'