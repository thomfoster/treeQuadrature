from .distributedTreentegrator import DistributedTreeIntegrator
from ..samplers import Sampler
from ..containerIntegration.gpIntegral import IterativeGpIntegral
from ..container import Container
from ..trees import Tree
from ..exampleProblems import Problem

from typing import Optional, List
import numpy as np

# parallel computing
from concurrent.futures import ProcessPoolExecutor, as_completed


def individual_container_integral(integral: IterativeGpIntegral, 
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
    previous_samples : dict
        containers are keys and values are 
        
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
    
    if not hasattr(integral, "n_samples"):
        raise AttributeError("integral must have attribute n_samples")
    
    integral.n_samples = n_samples

    n_previous = cont.N

    integral_results, new_samples = integral.containerIntegral(
        cont, integrand, return_std=return_std, 
        previous_samples=container_samples
    )

    if cont.N - n_previous != n_samples:
        raise RuntimeError("integral did not add proper number of samples. "
                           f"added {cont.N - n_previous} "
                           f"while expecting {n_samples}")

    # Return integral results, modified container, and the new samples
    return integral_results, cont, new_samples


class DistributedGpTreeIntegrator(DistributedTreeIntegrator):
    """
    Integrator for gaussian process container integrals
    that distribute samples dynamically based on performance gain.

    Attributes
    ---------
    split : Split
        method to split the tree
    integral : IterativeGpIntegral
        an integrator that takes previous_samples
        and is able to extend from there. \n
        Note: integral.n_samples will be ignored
    base_N : int
        number of initial samples used to construct the tree
    P : int
        stopping criteria for building tree. \n
        largest container should not have more than P samples
    sampler : Sampler
        sampler to use when problem does nove have method rvs
    max_container_samples : int
        maximum number of samples to allocate to a container
    min_container_samples : int
        minimum number of samples to allocate to a container. \n
        This is also used for the initial round
    max_iterations_per_container : int
        maximum number of iterations to allocate to a container
    """
    def __init__(self, base_N: int, max_n_samples: int, integral: IterativeGpIntegral,
                 tree: Optional[Tree]=None, 
                 sampler: Optional[Sampler]=None, 
                 max_container_samples: int=200, 
                 min_container_samples: int=20,
                 max_iterations_per_container: int = 5):
        super().__init__(base_N=base_N, integral=integral, sampler=sampler, tree=tree, max_n_samples=max_n_samples,
                         max_container_samples=max_container_samples, min_container_samples=min_container_samples)
        self.max_iterations_per_container: max_iterations_per_container
    
    def integrate_containers(self, containers: List[Container], 
                             problem: Problem,
                             compute_std: bool=False, 
                             verbose: bool=False):
        n_samples = np.sum([cont.N for cont in containers])

        # Initialize previous_samples dictionary
        previous_samples = {}
        container_iterations = {}
        container_performances = {}
        container_prev_performances = {}  # To store old performances
        total_samples = n_samples
        all_containers = []
        all_results = []

        sample_allocation = [self.min_container_samples for _ in range(len(containers))]
        if total_samples + sum(sample_allocation) > self.max_n_samples:
            raise ValueError('not enough samples to fit first run of GP'
                                'please reduce base_N or increase max_n_samples')

        while total_samples < self.max_n_samples and len(containers) > 0:
            if verbose:
                print(f"largest container allocation {max(sample_allocation)}")

            container_sample_map = {id(cont): sample_allocation[i] for i, cont in enumerate(containers)}
            if self.parallel:
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(individual_container_integral, 
                                        self.integral, cont, problem.integrand, 
                                        compute_std, container_sample_map[id(cont)],
                                        previous_samples=previous_samples): id(cont)
                        for cont in containers
                    }

                    results = []
                    new_samples_dict = {}
                    for future in as_completed(futures):
                        container_id = futures[future]
                        integral_results, container, new_samples = future.result()
                        results.append((integral_results, container))
                        new_samples_dict[container] = new_samples

                        total_samples += container_sample_map[container_id]
            else:
                results = []
                new_samples_dict = {}
                for cont in containers:
                    integral_results, container, new_samples = individual_container_integral(
                        self.integral, cont, problem.integrand, 
                        compute_std, container_sample_map[id(cont)],
                        previous_samples=previous_samples
                    )
                    results.append((integral_results, container))
                    new_samples_dict[container] = new_samples

                    total_samples += container_sample_map[id(cont)]

            if verbose:
                print(f"Total samples used: {total_samples}/{self.max_n_samples}")

            if total_samples >= self.max_n_samples:
                containers = [cont for _, cont in results]
                break

            # Update previous_samples with new samples from this iteration
            previous_samples.update(new_samples_dict)
            
            # Track performance gains
            for result, container in results:
                new_performance = result['performance']
                old_performance = container_prev_performances.get(container, new_performance)
                delta_performance = new_performance - old_performance if (
                    container in container_prev_performances) else new_performance
                container_prev_performances[container] = new_performance
                container_performances[container] = delta_performance
                container_iterations[container] = container_iterations.get(container, 0) + 1

            # Sort containers by performance gain (delta)
            ranked_containers_results = sorted(
                results, 
                key=lambda x: container_performances[x[1]],
                reverse=True
            )

            # Allocate samples dynamically based on performance gain
            available_samples = min(self.max_n_samples - total_samples, 
                                    self.min_container_samples * len(containers))
            sample_allocation = self._allocate_samples(ranked_containers_results=ranked_containers_results, 
                                                       available_samples=available_samples,
                                                       container_iterations=container_iterations,
                                                       max_iterations_per_container=self.max_iterations_per_container, 
                                                       container_performances=container_performances,
                                                       max_per_container=self.max_container_samples)
            
            # Separate out containers that received 0 samples
            containers_for_next_iteration = []
            updated_sample_allocation = []
            for idx, (result, container) in enumerate(ranked_containers_results):
                if sample_allocation[idx] > 0:
                    containers_for_next_iteration.append(container)
                    updated_sample_allocation.append(sample_allocation[idx])
                elif sample_allocation[idx] == 0:
                    all_containers.append(container)
                    all_results.append(result)
                else: 
                    raise RuntimeError("allocation cannot be negative, got "
                                       f"{sample_allocation[idx]}")

            containers = containers_for_next_iteration
            sample_allocation = updated_sample_allocation

            if verbose:
                print(f"Number of containers left: {len(containers)}")
        
        # Only add the remaining containers not yet processed
        all_containers.extend(containers)
        all_results.extend([result for result, _ in ranked_containers_results if result not in all_results])

    
    def _allocate_samples(self, ranked_containers_results: list, 
                      available_samples: int, max_per_container: int,
                      container_iterations: dict, 
                      max_iterations_per_container: int,
                      container_performances: dict) -> List[int]:
        """
        Allocate samples to containers based on their performance gain, with a strict cap on the number
        of samples allocated to any single container in this iteration.

        Parameters
        ----------
        ranked_containers_results : list
            A list of tuples where each tuple contains a result dictionary and a container,
            sorted by performance gain.
        available_samples : int
            The total number of samples available to be allocated.
        max_per_container : int, optional
            The maximum number of samples to allocate to any single container in this iteration.
            Defaults to 1000.
        container_iterations : dict
            Dictionary tracking the number of iterations each container has undergone.
        max_iterations_per_container : int
            The maximum number of iterations allowed per container.
        container_performances : dict
            Dictionary tracking the performance delta for each container.

        Returns
        -------
        List[int]
            A list containing the number of samples allocated to each container.
        """
        min_performance = min(container_performances.values(), default=0)

        # Ensure positive weights
        shift = -min_performance + 1e-10 if min_performance <= 0 else 0

        weights = []
        for result, container in ranked_containers_results:
            if container_iterations[container] >= max_iterations_per_container:
                weights.append(0)
                continue

            delta_performance = container_performances.get(container, 0) + shift
            # Use np.clip to ensure the log does not produce negative results
            weight = np.log(np.clip(delta_performance, 1e-10, None)) + 1
            weights.append(weight)

        weights = np.array(weights)
        total_weight = np.sum(weights)

        if total_weight > 0:
            normalized_weights = weights / total_weight
        else:
            # Uniform allocation if all weights are zero
            normalized_weights = np.ones_like(weights) / len(weights)

        allocation = np.floor(normalized_weights * available_samples).astype(int)
        allocated_samples = np.sum(allocation)
        discrepancy = available_samples - allocated_samples

        # Handle discrepancies by incrementally allocating or reducing samples
        if discrepancy > 0:
            while discrepancy > 0:
                for i in range(len(allocation)):
                    if discrepancy == 0:
                        break
                    if allocation[i] < max_per_container:
                        allocation[i] += 1
                        discrepancy -= 1
        elif discrepancy < 0:
            while discrepancy < 0:
                for i in range(len(allocation)):
                    if discrepancy == 0:
                        break
                    if allocation[i] > 0:
                        allocation[i] -= 1
                        discrepancy += 1

        # Ensure no over-allocation
        allocation = np.clip(allocation, 0, max_per_container)

        return allocation.tolist()