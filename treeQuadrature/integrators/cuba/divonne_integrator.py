from ..base_class import Integrator
from ...example_problems import Problem
from ...utils import ResultDict, integrand_for_cuba, compile_cuba_results

import pycuba


class DivonneIntegrator(Integrator):
    def __init__(self, degree: int=1,
                 partition_degree: int=1,
                 refinement_degree: int=1,
                 max_n_samples: int=None,
                 max_pass: int=5,
                 border: int=0.,
                 max_chisq: float=10.,
                 min_deviation: float=0.25
                 ):
        """
        Parameters
        ----------
        max_n_samples : int, Optional
            maximum number of evaluations. \n
            Default: 50,000
        degree : int, Optioanl
            the degree of cubature rule used
            in the main integration stage. \n
            Must be one of 7, 9, 11, 13. \n
            Otherwise, for positive degree,
            use Korobov quasi-random sample;
            for negative degree, use
            standard sample.
            Default: Korobov quasi-random sample.
        partition_degree : int, Optioanl
            the degree of cubature rule used
            in partitioning phase. \n
            same setting as degree
        refinement_degree : int, Optioanl
            the degree of cubature rule used
            in partitioning phase. \n
            0 = no refinement;
            1 = one more split;
            7, 9, 11, 13: further sampled using
            cubature rule of that degree. \n
            Otherwise, for positive degree,
            use Korobov quasi-random sample;
            for negative degree, use
            standard sample.
        max_pass : int, Optional
            maximum number of iterations for
            partition stage.
            Default: 5
        border : float, Optional
            the width of the border of
            the integration region. \n
	        Points falling into this border region
            are not sampled directly,
            but are extrapolated from
            two samples from the interior.
            Default: 0.0
        max_chisq : float, Optional
            maximum chi-square value a single subregion
            is allowed to have in the main integration phase.
            Default: 10.0
        min_deviation : float, Optional
            A value between 0 and 1. \n
            Regions which fail the chi-square test
            are not treated further
            if their sample averages differ by less
            than min_deviation.
        """
        self.degree = degree
        self.partition_degree = partition_degree
        self.refinement_degree = refinement_degree
        self.max_n_samples = max_n_samples
        self.max_pass = max_pass
        self.border = border
        self.max_chisq = max_chisq
        self.min_deviation = min_deviation

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
            passed to Divonne integrator.

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
        results = pycuba.Divonne(
            integrand_for_cuba(problem.integrand),
            ndim=problem.D, key1=self.partition_degree,
            key2=self.degree, key3=self.refinement_degree,
            maxeval=self.max_n_samples,
            maxpass=self.max_pass,
            border=self.border,
            maxchisq=self.max_chisq,
            mindeviation=self.min_deviation,
            **kwargs)
        
        resultDict = compile_cuba_results(results)

        return resultDict