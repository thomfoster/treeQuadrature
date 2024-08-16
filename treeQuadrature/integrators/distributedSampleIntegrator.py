from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import signature
import numpy as np
import warnings

from typing import Optional, Callable

from .simpleIntegrator import SimpleIntegrator
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler
from ..splits import Split
from ..exampleProblems import Problem
from ..container import Container


def parallel_container_integral(integral: ContainerIntegral, 
                                cont: Container, integrand: callable, 
                                return_std: bool, n_samples: int):
    try:
        integral.n_samples = n_samples
    except AttributeError:
        raise AttributeError("self.integral does not have attribute "
                             "n_samples, cannot use DistributedSampleIntegrator")
    
    params = {}
    if hasattr(integral, 'get_additional_params'):
        params.update(integral.get_additional_params())

    if return_std:
        params['return_std'] = True

    integral_results = integral.containerIntegral(cont, integrand, 
                                                    **params)
        
    # check types
    if 'integral' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'integral'")
    elif return_std and 'std' not in integral_results:
        raise KeyError("results of containerIntegral does not have key 'std'")
        
    return integral_results, cont

class DistributedSampleIntegrator(SimpleIntegrator):
    def __init__(self, base_N: int, P: int, max_n_samples: int, split: Split, 
                 integral: ContainerIntegral, 
                 sampler: Optional[Sampler]=None, 
                 construct_tree_method: Optional[Callable] = None):
        """
        An integrator that constructs a tree and then distributes the 
        remaining samples among the containers obtained 
        according to the volume of containers

        Parameters
        ----------
        base_N : int
            Total number of initial samples.
        P : int
            Maximum number of samples in each container during tree construction.
        max_n_samples : int
            Total number of evaluations available.
        split : Split
            Method to split a container during tree construction.
        integral : ContainerIntegral 
            Method to evaluate the integral of f on a container.
        sampler : Sampler, optional
            Method for generating initial samples, 
            when the problem does not have an rvs method.
        construct_tree_method : Callable, optional
            Custom method to construct the tree. If None, use the default 
            `construct_tree` method from `SimpleIntegrator`. 
        """
        super().__init__(base_N, P, split, integral, sampler)
        self.max_n_samples = max_n_samples
        self.construct_tree_method = construct_tree_method or super().construct_tree

    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, 
                 return_std: bool=False, verbose: bool=False,
                 return_all: bool=False, 
                 *args, **kwargs) -> dict:

        if verbose: 
            print('Drawing initial samples')
        # Draw samples
        if self.sampler is not None:
            X, y = self.sampler.rvs(self.base_N, problem.lows, problem.highs,
                                    problem.integrand)
        elif hasattr(problem, 'rvs'):
            X = problem.rvs(self.base_N)
            y = problem.integrand(X)
        else:
            raise RuntimeError('Cannot draw initial samples. '
                               'Either problem should have rvs method, '
                               'or specify self.sampler')

        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), (
            'The output of problem.integrand must be one-dimensional array'
            f', got shape {y.shape}'
        )

        if verbose: 
            print('Constructing root container')
        root = Container(X, y, mins=problem.lows, maxs=problem.highs)

        # Construct tree
        if verbose:
            print('Constructing tree')
            if 'verbose' in signature(self.construct_tree_method).parameters:
                finished_containers = self.construct_tree_method(root, verbose=True, 
                                                          *args, **kwargs)
            else:
                finished_containers = self.construct_tree_method(root, 
                                                          *args, **kwargs)
        else:
            finished_containers = self.construct_tree_method(root, *args, **kwargs)

        if len(finished_containers) == 0:
            raise RuntimeError('No container obtained from construct_tree')
        
        if verbose:
            n_samples = np.sum([cont.N for cont in finished_containers])
            print(f'Got {len(finished_containers)} containers with {n_samples} samples')

        # Determine the number of remaining samples to distribute
        used_samples = np.sum([cont.N for cont in finished_containers])
        remaining_samples = self.max_n_samples - used_samples
        print(f"remaining samples = {remaining_samples}")

        if remaining_samples > 0:
            total_volume = sum(c.volume for c in finished_containers)
            samples_distribution = {}

            # Initial distribution based on volume
            total_assigned = 0
            for i, cont in enumerate(finished_containers):
                additional_samples = max(1, int(remaining_samples * 
                                                (cont.volume / total_volume)))
                samples_distribution[cont] = additional_samples
                total_assigned += additional_samples

            # Distribute the remaining samples
            remainder_samples = remaining_samples - total_assigned
            while remainder_samples > 0:
                for cont in finished_containers:
                    if remainder_samples <= 0:
                        break
                    samples_distribution[cont] += 1
                    remainder_samples -= 1
        
        print(f"distributed samples: {sum(samples_distribution)}")


        # Uncertainty estimates
        method = getattr(self.integral, 'containerIntegral', None)
        if method:
            has_return_std = 'return_std' in signature(method).parameters
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

        # for retracking containers 
        containers = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(parallel_container_integral, 
                                self.integral, cont, problem.integrand, 
                                compute_std, n_samples=samples_distribution.get(cont)): cont
                for cont in finished_containers
            }

            results = []
            for future in as_completed(futures):
                integral_results, modified_cont = future.result()
                results.append(integral_results)
                containers.append(modified_cont)
        
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

        return_values = {'estimate': G}
        if return_N:
            return_values['n_evals'] = N
        if return_containers:
            return_values['containers'] = containers
            return_values['contributions'] = contributions
        if compute_std:
            return_values['stds'] = stds

        return return_values