from ..base_class import Integrator
from ...example_problems import Problem
from ...utils import ResultDict, integrand_for_cuba, compile_cuba_results

import pycuba


class CuhreIntegrator(Integrator):
    def __init__(self, degree: int=0,
                 max_n_samples: int=None):
        """
        Parameters
        ----------
        degree: int, Optioanl
            the degree of cubature rule used.
            Must be one of 7, 9, 11, 13. \n
            Default: degree-13 rule in 2 dimensions,
            degree-11 rule in 3 dimensions,
            degree-9 rule otherwise.
        max_n_samples: int, Optional
            maximum number of evaluations. \n
            Default: 50,000
        kwargs : Any
            passed to Cuhre integrator.
        """
        self.degree = degree
        self.max_n_samples = max_n_samples

    def __call__(self, problem: Problem,
                 return_N: bool=True,
                 **kwargs) -> ResultDict:
        """
        Parameters
        ----------
        prolem : Problem
            the integration problem being solved
        return_N : bool
            a placeholder,
            number of evaluations will
            always be included.

        Return
        ------
        ResultDict
            - 'estimate' (float): Estimated integral value.
            - 'n_evals' (int): Number of function evaluations
            - 'contributions' (list[float]): Contributions of each 
            region in estimate,
            - 'stds' (list[float]): Uncertainty estimates of the
            integral estimate in each region.
        """
        results = pycuba.Cuhre(
            integrand_for_cuba(problem.integrand),
            ndim=problem.D, key=self.degree,
            maxeval=self.max_n_samples, **kwargs)
        
        resultDict = compile_cuba_results(results)

        return resultDict