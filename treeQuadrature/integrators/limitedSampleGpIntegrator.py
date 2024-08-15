from ..splits import Split
from ..samplers import Sampler
from ..containerIntegration.gpIntegral import IterativeGpIntegral
from ..container import Container
from .integrator import Integrator
from ..exampleProblems import Problem

from typing import Optional, List
from queue import SimpleQueue
import numpy as np
import time, warnings

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed


def parallel_container_integral(integral: IterativeGpIntegral, 
                                cont: Container, integrand: callable, 
                                return_std: bool, n_samples: int, 
                                previous_samples: Optional[dict] = None):
    
    """
    Perform the container integral with the option to pass in previous samples.

    Parameters
    ----------
    integral : IterativeGpIntegral
        The integral method to be used.
    cont : Container
        The container to perform the integral on.
    integrand : callable
        The integrand function.
    return_std : bool
        Whether to return the standard deviation of the integral.
    n_samples : int
        Number of samples to draw for this container in this iteration.
    previous_samples : tuple of np.ndarray, optional
        Tuple containing previous samples (xs, ys) if available, otherwise None.
        
    Returns
    -------
    integral_results : dict
        The results of the integral.
    cont : Container
        The container used for the integral.
    new_samples : tuple of np.ndarray
        Updated tuple containing all samples used.
    """    
    if previous_samples is None:
        previous_samples = {}

    container_samples = previous_samples.get(cont, None)
    
    integral.n_samples = n_samples

    integral_results, new_samples = integral.containerIntegral(
        cont, integrand, return_std=return_std, 
        previous_samples=container_samples
    )

    # Return integral results, modified container, and the new samples
    return integral_results, cont, new_samples

class LimitedSamplesGpIntegrator(Integrator):
    """
    Integrator for gaussian process container integrals

    Attributes
    ---------
    split : Split
        method to split the tree
    integral : IterativeGpIntegral
        an integrator that takes previous_samples
        and is able to extend from there
    base_N : int
        number of initial samples used to construct the tree
    P : int
        stopping criteria for building tree
        largest container should not have more than P samples
    sampler : Sampler
        sampler to use when problem does nove have method rvs
    """
    def __init__(self, base_N: int, max_n_samples: int, P: int, split: Split, 
                 integral: IterativeGpIntegral,
                 sampler: Optional[Sampler]=None, 
                 max_container_samples: int=200):
        self.split = split
        self.base_N = base_N
        self.integral = integral
        self.P = P
        self.max_n_samples = max_n_samples
        self.sampler = sampler
        self.n_samples = integral.n_samples
        self.max_container_samples = max_container_samples

    def construct_tree(self, root: Container, verbose: bool=False, 
                       **kwargs) -> List[Container]:
        """
        Construct a simple tree 

        Arguments
        ---------
        root: Container
            the root container
        verbose : bool, optional (default = False)
            whether print queue status every 100 iterations 
            or not
        max_iter : float, optional (default = 1e4)
            maximum number of iterations 
        """
        max_iter = kwargs.get('max_iter', 1e4)
        tree_containers = []
        q = SimpleQueue()
        q.put(root)
        start_time = time.time()
        iteration_count = 0

        while not q.empty() and iteration_count < max_iter:
            iteration_count += 1
            c = q.get()
            if c.N <= self.P:
                tree_containers.append(c)
            else:
                children = self.split.split(c)
                for child in children:
                    q.put(child)
            
            if iteration_count % 100 == 0 and verbose:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.qsize()}, "
                      f"number of containers = {len(tree_containers)}, "
                      f"Elapsed time = {elapsed_time:.2f}s")

        if iteration_count == max_iter:
            warnings.warn('Maximum iterations reached for constructing the tree. '
                          'Increase max_iter or check split and samples.', RuntimeWarning)
            while not q.empty():
                tree_containers.append(q.get())

        return tree_containers

    def __call__(self, problem: Problem, 
                 return_N: bool=False, return_containers: bool=False, 
                 return_std: bool=False, verbose: bool=False,
                 return_all: bool=False, 
                 **kwargs) -> dict:
        """
        Perform the integration process.

        Parameters
        ----------
        problem : Problem
            The integration problem to be solved.
        return_N : bool, optional
            If true, return the number of function evaluations.
        return_containers : bool, optional
            If true, return containers and their contributions as well.
        return_std : bool, optional
            If true, return the standard deviation estimate. 
            Ignored if self.integral does not have return_std attribute.
        verbose : bool, optional
            If true, print the stages (for debugging).
        return_all : bool, optional
            If true, returns a list of raw results from self.integral.containerIntegral.
        **kwargs : Any
            additional arguments for constructing the tree 

        Returns
        -------
        dict
            Dictionary containing integration results.
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
        containers = self.construct_tree(root, verbose=verbose, 
                                                **kwargs)

        if len(containers) == 0:
            raise RuntimeError('No container obtained from construct_tree')
        
        n_samples = np.sum([cont.N for cont in containers])
        if verbose:
            print(f'got {len(containers)} containers with {n_samples} samples')

        # Initialize previous_samples dictionary
        previous_samples = {}

        total_samples = n_samples
        all_containers = []
        all_results = []

        sample_allocation = [self.n_samples for _ in range(len(containers))]
        if total_samples + sum(sample_allocation) > self.max_n_samples:
            raise ValueError('not enough samples to fit first run of GP'
                                'please reduce base_N or increase max_n_samples')

        while total_samples < self.max_n_samples and len(containers) > 0:
            total_samples += sum(sample_allocation)
            if verbose:
                print(f"largest container allocation {max(sample_allocation)}")

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(parallel_container_integral, 
                                    self.integral, cont, problem.integrand, 
                                    return_std, sample_allocation[i],
                                    previous_samples=previous_samples): cont
                    for i, cont in enumerate(containers)
                }

                results = []
                modified_containers = []
                new_samples_dict = {}
                for future in as_completed(futures):
                    integral_results, modified_cont, new_samples = future.result()
                    results.append(integral_results)
                    modified_containers.append(modified_cont)
                    new_samples_dict[modified_cont] = new_samples

            # Update previous_samples with new samples from this iteration
            previous_samples.update(new_samples_dict)

            ranked_containers_results = sorted(
                zip(results, modified_containers), 
                key=lambda x: x[0]['performance'], 
                reverse=self.integral.score_direction == 'down'
            )

            # Allocate samples dynamically based on GP performance
            # for next iteration
            sample_allocation = self._allocate_samples(ranked_containers_results, 
                                                       min(self.max_n_samples - total_samples, 
                                                           self.n_samples * len(containers)),
                                                       max_per_container=self.max_container_samples)
            
            # Separate out containers that received 0 samples
            containers_for_next_iteration = []
            for idx, (result, container) in enumerate(ranked_containers_results):
                if sample_allocation[idx] > 0:
                    containers_for_next_iteration.append(container)
                else:
                    all_containers.append(container)
                    all_results.append(result)

            containers = containers_for_next_iteration

            if verbose:
                print(f"Total samples used: {total_samples}/{self.max_n_samples}")
                print(f"Number of containers left: {len(containers)}")
        
        all_containers.extend([cont for _, cont in ranked_containers_results])
        all_results.extend([res for res, _ in ranked_containers_results])

        if len(all_containers) != len(all_results):
            raise RuntimeError(f'number of containers ({len(all_containers)}) not the same as '
                               f'numebr of integral results ({len(all_results)})')

        if return_all:
            return all_results, all_containers

        contributions = [result['integral'] for result in all_results]
        if return_std:
            stds = [result['std'] for result in all_results]
        else:
            stds = None

        G = np.sum(contributions)
        N = sum([cont.N for cont in all_containers])

        return_values = {'estimate' : G}
        if return_N:
            return_values['n_evals'] = N
        if return_containers:
            return_values['containers'] = all_containers
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = stds

        return return_values
    
    def _allocate_samples(self, ranked_containers_results: list, available_samples: int, max_per_container: int = 1000):
        """
        Allocate samples to containers based on their performance, with a cap on the number
        of samples allocated to any single container in this iteration.

        Parameters
        ----------
        ranked_containers_results : list
            A list of tuples where each tuple contains a result dictionary and a container,
            sorted by performance.
        available_samples : int
            The total number of samples available to be allocated.
        max_per_container : int, optional
            The maximum number of samples to allocate to any single container in this iteration.
            Defaults to 1000.

        Returns
        -------
        List[int]
            A list containing the number of samples allocated to each container.
        """
        performances = np.array([result['performance'] for result, _ in ranked_containers_results])

        # Invert the performance scores so that lower-performing containers get more samples
        inverted_performances = np.max(performances) - performances

        # Apply a softening function to the inverted performances to reduce aggressiveness
        softened_weights = np.power(inverted_performances + 1e-10, 0.5)

        # Normalize the softened weights to sum to 1
        allocation_weights = softened_weights / np.sum(softened_weights)

        # Allocate samples based on these weights, with a minimum allocation for each container
        min_allocation = max(1, available_samples // (2 * len(allocation_weights)))
        allocation = np.round(np.maximum(allocation_weights * available_samples, min_allocation)).astype(int)

        # Ensure the total allocation does not exceed available_samples
        total_allocated = np.sum(allocation)
        if total_allocated > available_samples:
            excess = total_allocated - available_samples
            while excess > 0:
                idx = np.argmax(allocation)
                reduction = min(excess, allocation[idx] - min_allocation)
                allocation[idx] -= reduction
                excess -= reduction

        # Adjust to ensure all available samples are allocated
        discrepancy = available_samples - np.sum(allocation)
        while discrepancy > 0:
            idx = np.argmax(allocation_weights)
            allocation[idx] += 1
            discrepancy -= 1

        return allocation.tolist()
