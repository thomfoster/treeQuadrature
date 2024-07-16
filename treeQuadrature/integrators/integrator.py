from abc import ABC, abstractmethod
from ..exampleProblems import Problem
from typing import Union, Tuple, Any

class Integrator(ABC):
    """
    Abstract base class for integrators.
    """

    @abstractmethod
    def __call__(self, problem: Problem) -> Union[float, Tuple[float, Any]]:
        """
        Perform integration on the given problem.

        Parameters
        ----------
        problem : Problem
            The problem to be integrated.

        Returns
        -------
        float or Tuple
            the first return value must be float 
              representing the estimated integral value
        """
        pass