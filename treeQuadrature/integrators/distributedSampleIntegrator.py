from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import signature
import numpy as np
import warnings

from typing import Optional, Callable, List

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
        _ = integral.n_samples
    except AttributeError:
        raise AttributeError("self.integral does not have attribute "
                             "n_samples, cannot use DistributedSampleIntegrator")

    integral.n_samples = n_samples
    
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
    def __init__(self, base_N: int, P: int, max_n_samples: int, 
                 split: Split, 
                 integral: ContainerIntegral, 
                 sampler: Optional[Sampler]=None, 
                 construct_tree_method: Optional[Callable[[Container], List[Container]]] = None,
                 scaling_factor: float = 1e-6,
                 min_container_samples: int = 2, 
                 max_container_samples: int = 200) -> None:
        """
        An integrator that constructs a tree and then distributes the 
        remaining samples among the containers obtained 
        according to the volume of containers.

        Parameters
        ----------
        base_N : int
            Total number of initial samples.
        P : int
            Maximum number of samples in each container during tree construction.
        max_n_samples : int
            Total number of evaluations available.
        min_container_samples, max_container_samples: int, optional
            The minimum and maximum number of samples to allocate to each container 
            Defaults are 2 and 200
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
        scaling_factor : float, optional
            A scaling factor to control the aggressiveness of sample distribution 
            (default is 1e-6).
        """
        super().__init__(base_N, P, split, integral, sampler)
        self.max_n_samples = max_n_samples
        self.max_container_samples = max_container_samples
        self.min_container_samples = min_container_samples
        self.construct_tree_method = construct_tree_method or super().construct_tree
        self.scaling_factor = scaling_factor

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
        
        n_samples = np.sum([cont.N for cont in finished_containers])
        if n_samples > self.base_N:
            raise RuntimeError('construct_tree_method uses more samples than base_N! ')
        
        if verbose:
            print(f'Got {len(finished_containers)} containers with {n_samples} samples')

        # Uncertainty estimates
        method = getattr(self.integral, 'containerIntegral', None)
        if method:
            has_return_std = 'return_std' in signature(method).parameters
        else:
            raise TypeError("self.integral must have 'containerIntegral' method")
        compute_std = return_std and has_return_std
        if not has_return_std and compute_std:
            warnings.warn(
                f'{str(self.integral)}.containerIntegral does not have '
                'parameter return_std, will be ignored', 
                UserWarning
            )
            compute_std = False
        
        # integrate containers
        results, containers = self.integrate_containers(finished_containers, problem, 
                                                        compute_std, verbose)
        
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
    
    def integrate_containers(self, containers: List[Container], 
                             problem: Problem,
                             compute_std: bool=False, 
                             verbose: bool=False):
        # Determine the number of remaining samples to distribute
        used_samples = np.sum([cont.N for cont in containers])
        remaining_samples = self.max_n_samples - used_samples

        if used_samples + len(containers) * self.min_container_samples > self.max_n_samples:
            raise RuntimeError("too many samples to distribute. "
                               "either decrease 'min_container_samples' "
                               "or increase 'max_n_samples'")

        if remaining_samples > 0:
            samples_distribution = self._distribute_samples(containers, 
                                           remaining_samples, 
                                           self.min_container_samples, 
                                           self.max_container_samples, 
                                           problem.D)

        total_samples = sum(samples_distribution.values())
        if total_samples > remaining_samples:
            raise RuntimeError("allocated too many samples"
                               f"upper limit: {remaining_samples}"
                               f"allocated samples: {total_samples}")

        if verbose: 
            print('Integrating individual containers', 
                'with standard deviation' if compute_std else '')
            print(f"largest container distribution: {max(samples_distribution.values())}")
            print(f"smallest container distribution: {min(samples_distribution.values())}")

        # for retracking containers 
        modified_containers = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(parallel_container_integral, 
                                self.integral, cont, problem.integrand, 
                                compute_std, n_samples=samples_distribution.get(cont)): cont
                for cont in containers
            }

            results = []
            for future in as_completed(futures):
                integral_results, modified_cont = future.result()
                results.append(integral_results)
                modified_containers.append(modified_cont)

        return results, modified_containers


    def _distribute_samples(self, finished_containers, remaining_samples, 
                            min_container_samples, max_container_samples, 
                            problem_dim):
        total_volume = sum(c.volume for c in finished_containers)
        samples_distribution = {}

        # Initial distribution based on scaled volume
        total_assigned = 0
        for cont in finished_containers:
            scaled_volume = (cont.volume / total_volume) ** (1 / problem_dim)
            # Calculate initial allocation
            additional_samples = max(min_container_samples, 
                                    int(remaining_samples * scaled_volume))
            # Cap at max_container_samples and ensure minimum allocation
            additional_samples = min(max_container_samples, additional_samples)
            
            samples_distribution[cont] = additional_samples
            total_assigned += additional_samples

        # Adjust allocations if too many samples were assigned
        if total_assigned > remaining_samples:
            excess = total_assigned - remaining_samples
            for cont in sorted(finished_containers, key=lambda c: c.volume):
                if excess <= 0:
                    break
                reduce_by = min(excess, samples_distribution[cont] - 
                                min_container_samples)
                samples_distribution[cont] -= reduce_by
                excess -= reduce_by

        # Re-check and distribute any leftover samples if less were assigned
        if total_assigned < remaining_samples:
            remainder_samples = remaining_samples - total_assigned
            for cont in sorted(finished_containers, key=lambda c: -c.volume):
                if remainder_samples <= 0:
                    break
                if samples_distribution[cont] < max_container_samples:
                    samples_to_add = min(remainder_samples, 
                                         max_container_samples - samples_distribution[cont])
                    samples_distribution[cont] += samples_to_add
                    remainder_samples -= samples_to_add

        # Final check to ensure we never allocate more than allowed
        if sum(samples_distribution.values()) > remaining_samples:
            raise RuntimeError("Allocated too many samples")

        return samples_distribution
