from ..splits import Split
from ..samplers import Sampler
from ..containerIntegration.gpIntegral import IterativeGpIntegral
from ..container import Container
from .integrator import Integrator
from ..exampleProblems import Problem

from typing import Optional, List, Tuple
from queue import SimpleQueue
import numpy as np
import time, warnings

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed


def parallel_container_integral(integral: IterativeGpIntegral, 
                                cont: Container, integrand: callable, return_std: bool, 
                                previous_samples: Optional[dict]=None):
    
    """
    Perform the container integral with the option to pass in previous samples.

    Parameters
    ----------
    integral : IterativeRbfIntegral
        The integral method to be used.
    cont : Container
        The container to perform the integral on.
    integrand : callable
        The integrand function.
    return_std : bool
        Whether to return the standard deviation of the integral.
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
    integral_results, new_samples = integral.containerIntegral(
        cont, integrand, return_std=return_std, 
        previous_samples=container_samples
    )
    previous_samples[cont] = new_samples

    # Validate results
    if 'integral' not in integral_results:
        raise KeyError("Results of containerIntegral do not have key 'integral'")
    if return_std and 'std' not in integral_results:
        raise KeyError("Results of containerIntegral do not have key 'std'")
        
    return integral_results, cont, previous_samples

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
                 sampler: Optional[Sampler]=None):
        self.split = split
        self.base_N = base_N
        self.integral = integral
        self.P = P
        self.max_n_samples = max_n_samples
        self.sampler = sampler
        self.n_samples = integral.n_samples

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
        
        if verbose:
            n_samples = np.sum([cont.N for cont in containers])
            print(f'got {len(containers)} containers with {n_samples} samples')

        # Initialize previous_samples dictionary
        previous_samples = {}

        total_samples = self.base_N
        all_containers = []
        all_results = []

        while total_samples < self.max_n_samples:
            requested_samples = (self.max_n_samples - total_samples
                                 ) + len(containers) * self.n_samples
            if total_samples == self.base_N and requested_samples > self.max_n_samples:
                raise ValueError('not enough samples to fit first run of GP'
                                 'please reduce base_N or increase max_n_samples')

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(parallel_container_integral, 
                                    self.integral, cont, problem.integrand, 
                                    return_std, previous_samples=previous_samples): cont
                    for cont in containers
                }

                results = []
                modified_containers = []
                for future in as_completed(futures):
                    integral_results, modified_cont, previous_samples = future.result()
                    results.append(integral_results)
                    modified_containers.append(modified_cont)

            ranked_containers_results = sorted(
                zip(results, modified_containers), 
                key=lambda x: x[0]['performance'], 
                reverse=self.integral.score_direction == 'down'
            )
            total_samples += len(containers) * self.n_samples
            additional_samples = self.max_n_samples - total_samples
            num_poor_containers = min(int(np.floor(additional_samples / self.integral.n_samples)), 
                                      len(ranked_containers_results))
            poor_containers = [cont for _, cont in ranked_containers_results[:num_poor_containers]]
            good_containers = [cont for _, cont in ranked_containers_results[num_poor_containers:]]
            poor_results = [res for res, _ in ranked_containers_results[:num_poor_containers]]
            good_results = [res for res, _ in ranked_containers_results[num_poor_containers:]]

            all_results.extend(good_results)
            all_containers.extend(good_containers)
            containers = poor_containers

            if verbose:
                print(f"Total samples used: {total_samples}/{self.max_n_samples}")

            if len(poor_containers) == 0:
                break
        
        all_containers.extend(poor_containers)
        all_results.extend(poor_results)

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
