from abc import ABC, abstractmethod
from ..exampleProblems import Problem
from typing import Union, Tuple, Any

class Integrator(ABC):
    """
    Abstract base class for integrators.
    """

    @abstractmethod
    def __call__(self, problem: Problem, return_N: bool, **kwargs: Any) -> dict:
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
            estimate : float
                estimated integral value
            n_evals : int
                number of function estiamtions
            and other necessary details
        """
        pass

    # def test_call_method(self):
    #     """
    #     Test the __call__ method to ensure it returns a tuple 
    #     """
    #     result = self.__call__()
    #     if not isinstance(result, tuple):
    #         raise AssertionError("The result should be a tuple.")
        
    #     if not isinstance(result[0], dict):
    #         raise AssertionError("The first element of the tuple should be a dictionary.")
        
    #     if 'estimate' not in result[0]:
    #         raise AssertionError("The dictionary should contain the key 'estimate'.")
        
    #     if not isinstance(result[0]['estimate'], float):
    #         raise AssertionError("The value of 'estimate' should be a float.")