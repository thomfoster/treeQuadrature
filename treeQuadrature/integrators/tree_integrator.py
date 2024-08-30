from typing import List, Optional
from inspect import signature

import warnings
import numpy as np

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base_class import Integrator
from ..container import Container
from ..trees import Tree, SimpleTree
from ..container_integrators import ContainerIntegral, RandomIntegral
from ..samplers import Sampler
from ..example_problems import Problem
from ..utils import ResultDict


def integral_wrapper(
    integral: ContainerIntegral, cont: Container,
    integrand: callable, return_std: bool
):
    """
    Perform integration on an individual container.

    Parameters
    ----------
    integral : ContainerIntegral
        The integral method to be used.
    cont : Container
        The container on which to perform the integration.
    integrand : callable
        The integrand function to be integrated.
    return_std : bool
        Whether to return the standard deviation of the integral.

    Returns
    -------
    dict
        A dictionary containing the results of the integration.
        - 'estimate' : the estimated integral value
        - 'std' : the standard deviation of the integral value
    """
    params = {}
    if hasattr(integral, "get_additional_params"):
        params.update(integral.get_additional_params())

    if return_std:
        params["return_std"] = True
        integral_results = integral.containerIntegral(
            cont, integrand, **params)
    else:
        integral_results = integral.containerIntegral(
            cont, integrand, **params)
    # check results
    if "integral" not in integral_results:
        raise KeyError(
            "results of containerIntegral does not have key 'integral'"
            )
    elif return_std and "std" not in integral_results:
        raise KeyError(
            "results of containerIntegral does not have key 'std'"
            )

    return integral_results, cont


class TreeIntegrator(Integrator):
    def __init__(
        self,
        base_N: int,
        tree: Optional[Tree] = None,
        integral: Optional[ContainerIntegral] = None,
        sampler: Optional[Sampler] = None,
        parallel: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialise the TreeIntegrator.

        Parameters
        ----------
        base_N : int
            The number of initial samples to draw.
        tree : Tree, Optional
            The tree to be used for the integration process.
            Defaults to SimpleTree.
        integral : ContainerIntegral, Optional
            The integral method to be used on each container.
            Defaults to RandomIntegral.
        sampler : Sampler, Optional
            The sampler to be used to draw initial samples.
            If None, problem.rvs will be used.
        parallel : bool, Optional
            Whether to use parallel computing.
            Defaults to True.
        *args, **kwargs : Any
            Additional arguments to be passed to the tree construction method.

        Example
        -------
        >>> from treeQuadrature.integrators import TreeIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.example_problems import SimpleGaussian
        >>> from treeQuadrature.trees import WeightedTree
        >>> # Define the problem
        >>> problem = SimpleGaussian(D=2)
        >>> volume_weighting = lambda container: container.volume
        >>> stopping_small_containers = lambda container: container.N < 2
        >>> tree = WeightedTree(split=MinSseSplit(), max_splits=200,
        >>>     weighting_function=volume_weighting,
        >>>     stopping_condition=stopping_small_containers)
        >>> # Combine all compartments into a TreeIntegrator
        >>> integ_weighted = TreeIntegrator(base_N=1000, tree=tree,
        >>>                                 integral=RandomIntegral())
        >>> estimate = integ_weighted(problem)
        >>> print("error of random integral =",
        >>>       str(
        >>>         100 * np.abs(estimate - problem.answer) / problem.answer),
        >>>       "%")
        """
        super().__init__(*args, **kwargs)
        self.tree = tree if tree is not None else SimpleTree()
        self.integral = integral if integral is not None else RandomIntegral()
        self.base_N = base_N
        self.sampler = sampler
        self.parallel = parallel

    def __call__(
        self,
        problem: Problem,
        return_N: bool = False,
        return_containers: bool = False,
        return_std: bool = False,
        verbose: bool = False,
        return_all: bool = False,
        *args,
        **kwargs,
    ) -> ResultDict:
        """
        Perform the integration process.

        Arguments
        ----------
        problem : Problem
            The integration problem to be solved
        return_N : bool
            if true, return the number of function evaluations
        return_containers : bool
            if true, return containers and their contributions as well
        return_std : bool
            if true, return the standard deviation estimate.
            Ignored if self.integral does not have return_std attribute
        verbose: bool, Optional
            if true, print the stages (for debugging)
            Defaults to False
        *args, **kwargs : Any
            Additional arguments to be passed to the tree construction method

        Return
        -------
        ResultDict
            - 'estimate' (float) : estimated integral value
            - 'n_evals' (int) :  number of function estiamtions,
              if return_N is True
            - 'containers' (list[Container]) : list of Containers,
              if return_containers is True
            - 'contribtions' (list[float]) : contributions of each
              container in estimate, if return_containers is True
            - 'stds' (list[float]) : standard deviation of the
              integral estimate in each container, if return_std is True
        list[dict], list[Container]
            if return_all, returns a list of raw results
            from self.integral.containerIntegral

        Example
        -------
        >>> # You can use settings othe than default in the following way
        >>> from treeQuadrature.integrators import TreeIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.example_problems import SimpleGaussian
        >>> from treeQuadrature.trees import WeightedTree
        >>> # Define the problem
        >>> problem = SimpleGaussian(D=2)
        >>> # Define the a tree splitting containers with larger volume first
        >>> minSseSplit = MinSseSplit()
        >>> volume_weighting = lambda container: container.volume
        >>> stopping_small_containers = lambda container: container.N < 2
        >>> tree = WeightedTree(split=minSseSplit, max_splits=50,
        >>>     weighting_function=volume_weighting,
        >>>     stopping_condition=stopping_small_containers)
        >>> # Combine all compartments into a TreeIntegrator
        >>> integ_weighted = TreeIntegrator(
        >>>     base_N=1000, tree=tree, integral=RandomIntegral())
        >>> estimate = integ_weighted(problem)
        >>> print("error of random integral =",
        >>>       str(
        >>>         100 * np.abs(estimate - problem.answer) / problem.answer),
        >>>       "%")
        """
        X, y = self._draw_initial_samples(
            problem, verbose)
        root = self._construct_root_container(
            X, y, problem, verbose)

        leaf_containers = self._construct_tree(
            root, problem, verbose, *args, **kwargs)

        compute_std = self._check_return_std(return_std)

        results, containers = self.integrate_containers(
            leaf_containers, problem, compute_std, verbose
        )

        return self._compile_results(
            results, containers, compute_std,
            return_N, return_containers, return_all
        )

    def _draw_initial_samples(self, problem: Problem, verbose: bool):
        if verbose:
            print("drawing initial samples")
        if self.sampler is not None:
            X, y = self.sampler.rvs(
                self.base_N,
                problem.lows, problem.highs,
                problem.integrand
            )
        elif hasattr(problem, "rvs"):
            X = problem.rvs(self.base_N)
            y = problem.integrand(X)
        else:
            raise RuntimeError(
                "Cannot draw initial samples. "
                "Either problem should have rvs method, "
                "or specify self.sampler"
            )
        assert y.ndim == 1 or (
            y.ndim == 2 and y.shape[1] == 1
        ), "The output of problem.integrand must be " + \
           "a one-dimensional array, got shape {y.shape}"
        return X, y

    def _construct_root_container(
        self, X, y, problem: Problem, verbose: bool
    ) -> Container:
        if verbose:
            print("constructing root container")
        return Container(X, y, mins=problem.lows, maxs=problem.highs)

    def _construct_tree(
        self, root: Container, problem: Problem,
        verbose: bool, *args, **kwargs
    ) -> List[Container]:
        if verbose:
            print("constructing tree")
        construct_tree_parameters = signature(
            self.tree.construct_tree).parameters
        construct_tree_kwargs = {
            k: v for k, v in kwargs.items()
            if k in construct_tree_parameters
        }
        if "verbose" in construct_tree_parameters:
            construct_tree_kwargs["verbose"] = verbose
        if "integrand" in construct_tree_parameters:
            construct_tree_kwargs["integrand"] = problem.integrand

        finished_containers = self.tree.construct_tree(
            root, *args, **construct_tree_kwargs
        )

        if len(finished_containers) == 0:
            raise RuntimeError("No container obtained from construct_tree")

        if verbose:
            n_samples = np.sum([cont.N for cont in finished_containers])
            print(
                f"got {len(finished_containers)} containers "
                f"with {n_samples} samples"
            )

        return finished_containers

    def _check_return_std(self, return_std: bool) -> bool:
        method = getattr(self.integral, "containerIntegral", None)
        if method:
            has_return_std = "return_std" in signature(method).parameters
        else:
            raise TypeError(
                "self.integral must have 'containerIntegral' method")
        if return_std and not has_return_std:
            warnings.warn(
                f"{str(self.integral)}.containerIntegral does not have "
                "parameter return_std, will be ignored",
                UserWarning,
            )
            return False
        return return_std and has_return_std

    def _compile_results(
        self, results, containers, compute_std, return_N,
        return_containers, return_all
    ):
        if return_all:
            return results, containers

        contributions = [result["integral"] for result in results]
        G = np.sum(contributions)

        result_dict = ResultDict(estimate=G)

        if compute_std:
            stds = [result["std"] for result in results]
            result_dict["stds"] = stds

        if return_N:
            N = sum([cont.N for cont in containers])
            result_dict["n_evals"] = N
            if hasattr(self.tree, "n_splits"):
                result_dict["n_splits"] = self.tree.n_splits

        if return_containers:
            result_dict["containers"] = containers
            result_dict["contributions"] = contributions

        if compute_std:
            stds = [result["std"] for result in results]
            result_dict["stds"] = stds

        return result_dict

    def integrate_containers(
        self,
        containers: List[Container],
        problem: Problem,
        compute_std: bool = False,
        verbose: bool = False,
    ):
        if verbose:
            print(f"integrating containers, parallel : {self.parallel}")

        if len(containers) == 0:
            raise ValueError("Got no container")

        # for retracking containers
        modified_containers = []
        results = []

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        integral_wrapper,
                        self.integral,
                        cont,
                        problem.integrand,
                        compute_std,
                    ): cont
                    for cont in containers
                }

                for future in as_completed(futures):
                    integral_results, modified_cont = future.result()
                    results.append(integral_results)
                    modified_containers.append(modified_cont)
        else:
            for cont in containers:
                integral_results, modified_cont = integral_wrapper(
                    self.integral, cont, problem.integrand, compute_std
                )
                results.append(integral_results)
                modified_containers.append(modified_cont)

        return results, modified_containers
