from abc import abstractmethod
from typing import List, Optional
from inspect import signature

import warnings
import numpy as np

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .integrator import Integrator
from ..container import Container
from ..exampleProblems import Problem
from ..splits import Split
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler
from ..visualisation import plotContainers



def parallel_container_integral(integral: ContainerIntegral, 
                                cont: Container, integrand: callable, 
                                return_std: bool):
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
    # check types
    if 'integral' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'integral'")
    elif return_std and 'std' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'std'")
        
    return integral_results, cont

class TreeIntegrator(Integrator):
    """
    Abstract base class for tree integrators.
    """

    def __init__(self, split: Split,
            integral: ContainerIntegral, base_N: int, 
            sampler: Optional[Sampler]=None,
            *args, **kwargs):
        """
        Initialise the tree structure. 

        Arguments
        ---------
        split : Split
            the method to split the containers
        integral : ContainerIntegral
            the method to integrate the containers
        base_N : int
            number of initial samples
        sampler : Sampler
            a method for generating initial samples
            when problem does not have rvs method
        *args, **kwargs : Any
            other arguments necessary to build the Integrator,
            override __init__ in subclass to add other arguments
        """
        self.split = split
        self.integral = integral
        self.base_N = base_N
        self.sampler = sampler

    @abstractmethod
    def construct_tree(self, root: Container, *args, **kwargs) -> List[Container]:
        """
        Construct the tree based on a root container

        Arguments
        ---------
        root : Container
            the tree root
        integrand : the function to be 

        Return
        ------
        list of Containers
            leaf nodes of the tree
        """
        pass

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
            for construct_tree, 
            override __call__ in subclass to add additional arguments

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
        """

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

        # construct tree
        if verbose:
            print('constructing tree')
            if 'verbose' in signature(self.construct_tree).parameters:
                finished_containers = self.construct_tree(root, verbose=True, 
                                                          *args, **kwargs)
            else:
                finished_containers = self.construct_tree(root, 
                                                          *args, **kwargs)
        else:
            finished_containers = self.construct_tree(root, *args, **kwargs)

        if len(finished_containers) == 0:
            raise RuntimeError('No container obtained from construct_tree')
        
        if verbose:
            n_samples = np.sum([cont.N for cont in finished_containers])
            print(f'got {len(finished_containers)} containers with {n_samples} samples')

        # uncertainty estimates
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
 
        if verbose: 
            print('Integrating individual containers', 
                'with standard deviation' if compute_std else '')
        
        results, containers = self.integrate_containers(finished_containers, 
                                                        problem,
                                                        compute_std)
        
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
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(parallel_container_integral, 
                                self.integral, cont, problem.integrand, 
                                compute_std): cont
                for cont in containers
            }

            results = []
            for future in as_completed(futures):
                integral_results, modified_cont = future.result()
                results.append(integral_results)
                modified_containers.append(modified_cont)

        return results, modified_containers