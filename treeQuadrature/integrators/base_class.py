from abc import ABC, abstractmethod
from typing import Any

from ..example_problems import Problem
from ..utils import ResultDict


class Integrator(ABC):
    """
    Abstract base class for integrators.
    """

    @abstractmethod
    def __call__(self, problem: Problem, return_N: bool, **kwargs: Any) -> ResultDict:
        """
        Perform integration on the given problem.

        Parameters
        ----------
        problem : Problem
            The problem to be integrated.
        return_N : bool
            whether return number of evaluations or not

        Returns
        -------
        dict
            - estimate (float) : estimated integral value
            - n_evals (int) : number of function estiamtions
            - other necessary details
        """
        pass
