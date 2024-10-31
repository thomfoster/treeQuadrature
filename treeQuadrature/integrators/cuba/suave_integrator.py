from ..base_class import Integrator
from ...example_problems import Problem
from ...utils import ResultDict, integrand_for_cuba, compile_cuba_results

import pycuba


class SuaveIntegrator(Integrator):
    def __init__(self, flatness: float=50., n_new: int=1000,
                 n_min: int=2, max_n_samples: int=None):
        """
        Parameters
        ----------
        flatness: float, Optional
            the order of norm used when evaluating the
            uncertainty of subdivisions. \n
            A flatter integrand should use larger flatness.
            Defaults to 50.0
        n_new: int, Optional
            number of new integrand evaluations
            in each subdivision. \n
            Defaults to 1000.
        n_min: int, Optional
            minimum number of samples a former pass
            must contribute to a subregion.
        max_n_samples: int, Optional
            maximum number of evaluations. \n
            Default: 50,000
        """
        self.flatness = flatness
        self.n_new = n_new
        self.n_min = n_min
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
        kwargs : Any
            passed to Suave integrator.

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
        results = pycuba.Suave(
            integrand_for_cuba(problem.integrand),
            ndim=problem.D,
            nnew=self.n_new, nmin=self.n_min,
            flatness=self.flatness,
            maxeval=self.max_n_samples,
            **kwargs)
        
        resultDict = compile_cuba_results(results)

        return resultDict