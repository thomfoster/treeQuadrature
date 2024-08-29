from abc import abstractmethod
from typing import List, Optional
from inspect import signature

import warnings
import numpy as np

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .integrator import Integrator
from ..container import Container
from ..trees import Tree, SimpleTree
from ..containerIntegration import ContainerIntegral, RandomIntegral
from ..samplers import Sampler
from ..exampleProblems import Problem



def individual_container_integral(integral: ContainerIntegral, 
                                cont: Container, integrand: callable, 
                                return_std: bool):
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
        - 'integral' : the estimated integral value
        - 'std' : the standard deviation of the integral value
    """
    params = {}
    if hasattr(integral, 'get_additional_params'):
        params.update(integral.get_additional_params())

    if return_std:
        params['return_std'] = True
        integral_results = integral.containerIntegral(cont, integrand, 
                                                      **params)
    else: 
        integral_results = integral.containerIntegral(cont, integrand,
                                                      **params)
    # check results
    if 'integral' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'integral'")
    elif return_std and 'std' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'std'")
        
    return integral_results, cont

class TreeIntegrator(Integrator):
    """
    Tree-based integrator. 
    """

    def __init__(self, base_N: int, 
            tree: Optional[Tree]=None,
            integral: Optional[ContainerIntegral]=None, 
            sampler: Optional[Sampler]=None, 
            parallel: bool=True, 
            *args, **kwargs):
        """
        Initialise the tree structure. 

        Arguments
        ---------
        integral : ContainerIntegral, optional
            the method to integrate the containers
            default is RandomIntegral (mean of uniform samples redrawn)
        base_N : int
            number of initial samples for tree construction
        tree : Tree, optional
            the tree structure to use,
            default is SimpleTree
        sampler : Sampler, optional
            a method for generating initial samples
            when problem does not have rvs method. 
        parallel : bool, optional
            whether to use parallel computing for container integration
        """
        super().__init__(*args, **kwargs)
        self.tree = tree if tree is not None else SimpleTree()
        self.integral = integral if integral is not None else RandomIntegral()
        self.base_N = base_N
        self.sampler = sampler
        self.parallel = parallel

    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, 
                 return_std: bool=False, verbose: bool=False,
                 return_all: bool=False, 
                 *args, **kwargs) -> dict:
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
            for self.tree.construct_tree or self.integrate_containers

        Return
        -------
        dict
            with the following keys:
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
        >>> from treeQuadrature.exampleProblems import SimpleGaussian
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
        >>> integ_weighted = TreeIntegrator(base_N=1000, tree=tree, integral=RandomIntegral())
        >>> estimate = integ_weighted(problem)
        >>> print("error of random integral =", 
        >>>       str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
        """

        ### Draw initial samples
        if verbose: 
            print('drawing initial samples')
        # Draw samples
        if self.sampler is not None:
            X, y = self.sampler.rvs(self.base_N, problem.lows, problem.highs,
                                    problem.integrand)
        elif hasattr(problem, 'rvs'):
            X = problem.rvs(self.base_N)
            y = problem.integrand(X)
        else:
            raise RuntimeError('cannot draw initial samples. '
                               'Either problem should have rvs method, '
                               'or specify self.sampler'
                               )
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), (
            'the output of problem.integrand must be one-dimensional array'
            f', got shape {y.shape}'
        )

        if verbose: 
            print('constructing root container')
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        ### construct tree
        if verbose:
            print('constructing tree')
        construct_tree_parameters = signature(self.tree.construct_tree).parameters
        construct_tree_kwargs = {k: v for k, v in kwargs.items() if k in construct_tree_parameters}
        if 'verbose' in construct_tree_parameters:
            construct_tree_kwargs['verbose'] = verbose
        if 'integrand' in construct_tree_parameters:
            construct_tree_kwargs['integrand'] = problem.integrand
            
        finished_containers = self.tree.construct_tree(root, *args, **construct_tree_kwargs)

        if len(finished_containers) == 0:
            raise RuntimeError('No container obtained from construct_tree')
        
        if verbose:
            n_samples = np.sum([cont.N for cont in finished_containers])
            print(f'got {len(finished_containers)} containers with {n_samples} samples')

        ### check if self.integral has return_std attribute
        method = getattr(self.integral, 'containerIntegral', None)
        if method:
            has_return_std =  'return_std' in signature(method).parameters
        else:
            raise TypeError("self.integral must have 'containerIntegral' method")
        compute_std = return_std and has_return_std
        if not has_return_std and return_std:
            warnings.warn(
                f'{str(self.integral)}.containerIntegral does not have '
                'parameter return_std, will be ignored', 
                UserWarning
            )
            compute_std = False
 
        ### integrate containers
        if verbose: 
            print('Integrating individual containers', 
                'with standard deviation' if compute_std else '')
        
        # Check if integrate_containers accepts additional arguments
        integrate_containers_params = signature(self.integrate_containers).parameters
        integrate_kwargs = {k: v for k, v in kwargs.items() if k in integrate_containers_params}
        if 'verbose' in integrate_containers_params:
            integrate_kwargs['verbose'] = verbose
        results, containers = self.integrate_containers(finished_containers, 
                                                        problem,
                                                        compute_std, 
                                                        **integrate_kwargs)
        
        ### return results
        if return_all:
            return results, containers

        if compute_std:
            contributions = [result['integral'] for result in results]
            stds = [result['std'] for result in results]
        else:
            contributions = [result['integral'] for result in results]
            stds = None

        G = np.sum(contributions)
        N = sum([cont.N for cont in containers])


        return_values = {'estimate' : G}
        if return_N:
            return_values['n_evals'] = N
            if hasattr(self.tree, 'n_splits'): # actual number of splits when constructing the tree
                return_values['n_splits'] = self.tree.n_splits
        if return_containers:
            return_values['containers'] = containers
            return_values['contributions'] = contributions
        if compute_std:
            return_values['stds'] = stds

        return return_values

    def integrate_containers(self, containers: List[Container], 
                             problem: Problem,
                             compute_std: bool=False):
        if len(containers) == 0:
            raise ValueError("Got no container")

        # for retracking containers 
        modified_containers = []
        results = []

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(individual_container_integral, 
                                    self.integral, cont, problem.integrand, 
                                    compute_std): cont
                    for cont in containers
                }

                for future in as_completed(futures):
                    integral_results, modified_cont = future.result()
                    results.append(integral_results)
                    modified_containers.append(modified_cont)
        else:
            for cont in containers:
                integral_results, modified_cont = individual_container_integral(
                    self.integral, cont, problem.integrand, compute_std)
                results.append(integral_results)
                modified_containers.append(modified_cont)

        return results, modified_containers