from typing import List, Optional, Tuple
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..trees import Tree
from .tree_integrator import TreeIntegrator
from ..container_integrators import ContainerIntegral
from ..samplers import Sampler
from ..container import Container
from ..example_problems import Problem
from ..gaussian_process import kernel_integration, IterativeGPFitting
from ..utils import ResultDict



def build_grid(containers: List[Container], grid_size: int) -> dict:
    grid = {}
    for container in containers:
        mins = container.mins
        maxs = container.maxs
        grid_key = tuple(((mins + maxs) / 2 // grid_size).astype(int))
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(container)
    return grid

def find_neighbors_grid(grid: dict, container: Container, grid_size: int) -> List[Container]:
    neighbors = []
    mins = container.mins
    maxs = container.maxs
    grid_key = tuple(((mins + maxs) / 2 // grid_size).astype(int))
    for offset in np.ndindex((3,) * len(grid_key)):
        neighbor_key = tuple(np.array(grid_key) + np.array(offset) - 1)
        if neighbor_key in grid:
            for neighbor in grid[neighbor_key]:
                if neighbor != container:
                    neighbors.append(neighbor)
    return neighbors



class BatchGpIntegrator(TreeIntegrator):
    def __init__(self, base_N: int, integral: ContainerIntegral, 
                 tree: Optional[Tree] = None, 
                 base_grid_scale: float = 0.1, dimension_scaling_exponent: float = 0.9,
                 length_scaling_exponent: float = 0.1, max_n_samples: Optional[int] = None,
                 sampler: Optional[Sampler] = None):
        '''
        An integrator that allows communications between nodes
        for better GP tuning and fitting

        Attributes
        ----------
        base_N : int
            total number of initial samples
        tree : Tree
            a tree structure to split containers
        integral : ContainerIntegral 
            a method to evaluate the integral of f on a container. \n
            Must be GP integral and have return_hyper_params option. 
        sampler : Sampler
            a method for generating initial samples
            when problem does not have rvs method. \n
            Default: UniformSampler
        max_n_samples : int
            if set, samples will be distributed 
            evenly across batches
            until max_n_samples samples are used
        base_grid_scale : float, optional 
            the baseline scale for the grid size 
            before any adjustments for dimensionality or domain volume.
        dimension_scaling_exponent : float, optional
            Exponent to scale grid size with dimension.
        length_scaling_exponent : float, optional
            Exponent to scale grid size with average side length.

        
        Methods
        -------
        __call__(problem, return_N, return_all)
            solves the problem given and returns the estimate
        '''
        super().__init__(base_N=base_N, tree=tree, integral=integral, sampler=sampler)
        self.base_grid_scale = base_grid_scale
        self.dimension_scaling_exponent = dimension_scaling_exponent
        self.length_scaling_exponent = length_scaling_exponent
        self.max_n_samples = max_n_samples
        self.integral_results = {}

    def __call__(self, problem: Problem, return_N: bool = False, 
                 return_containers: bool = False, return_std: bool = False, 
                 verbose: bool = False, 
                 *args, **kwargs) -> ResultDict:
        """
        Perform the batch Gaussian process integration.

        Parameters:
        problem : Problem
            The problem to be integrated.
        return_N : bool, optional
            Whether to return the number of samples in each leaf container. \n
            Defaults to False.
        return_containers : bool, optional
            Whether to return the leaf containers. Defaults to False.
        return_std : bool, optional 
            Whether to return the standard deviation of the predictions. \n
              Defaults to False.
        verbose : bool, optional
            Whether to print verbose output. \n
            Defaults to False.
        *args, **kwargs
            Additional arguments to be passed to the build_tree method.

        Returns
        -------
        dict
            A dictionary containing the results of the integration.
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
        
        self._validate_inputs(problem)

        X, y = self._draw_initial_samples(problem, verbose)
        root = self._construct_root_container(X, y, problem, verbose)

        grid_size = self._construct_grid_size(problem, verbose)

        leaf_containers = self._build_tree_and_fit_gp(root, problem, grid_size, verbose, return_std, *args, **kwargs)

        return self._collect_results(leaf_containers, return_N, return_containers, return_std)

    def _validate_inputs(self, problem: Problem):
        if not isinstance(problem, Problem):
            raise ValueError("The input problem must be an instance of the Problem class.")

    def _construct_grid_size(self, problem: Problem, verbose: bool) -> float:
        avg_side_length = np.mean(problem.highs - problem.lows)
        grid_size = self.base_grid_scale * (
            problem.D ** self.dimension_scaling_exponent) * (
            avg_side_length ** (-self.length_scaling_exponent))
        
        if verbose:
            print(f"adaptive grid size: {grid_size}, average side length: {avg_side_length}")
        
        return grid_size

    def _build_tree_and_fit_gp(self, root: Container, problem: Problem, grid_size: float, 
                               verbose: bool, return_std: bool, *args, **kwargs) -> List[Container]:
        leaf_containers = self._construct_tree(root, problem, verbose, *args, **kwargs)

        if verbose:
            print('fitting GP to containers and passing hyper-parameters')
        self.integral._initialize_gp()
        self.fit_gps(leaf_containers, problem.integrand, verbose, return_std, grid_size)

        return leaf_containers

    def _collect_results(self, leaf_containers: List[Container], return_N: bool, 
                         return_containers: bool, return_std: bool) -> ResultDict:
        contributions = [self.integral_results[container]['integral'] for container in leaf_containers]

        return_values = ResultDict(estimate=np.sum(contributions))
        if return_N:
            return_values['n_evals'] = sum([container.N for container in leaf_containers])
        if return_containers:
            return_values['containers'] = leaf_containers
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = [self.integral_results[container]['std'] for container in leaf_containers]

        return return_values

    def fit_gps(self, containers: List[Container], integrand, verbose: bool = False, 
                return_std: bool = False, grid_size: float = 0.05):
        grid = self._build_batches(containers, grid_size)
        
        batches = [batch for _, batch in grid.items()]
        samples_per_batch, extra_samples = self._distribute_samples(batches)

        with ThreadPoolExecutor() as executor:
            if samples_per_batch:
                futures = [
                    executor.submit(self._fit_batch_gp, batch, integrand, samples_per_batch + (
                        1 if i < extra_samples else 0), verbose, 
                        grid_size, return_std)
                    for i, batch in enumerate(batches)
                ]
            else: # not set number of samples
                futures = [
                    executor.submit(self._fit_batch_gp, batch, integrand, None, verbose, 
                                    grid_size, return_std)
                    for batch in batches
                ]
            for future in futures:
                future.result()

    def _build_batches(self, containers: List[Container], grid_size: float) -> dict:
        return build_grid(containers, grid_size)

    def _distribute_samples(self, batches: List[List[Container]]) -> Tuple[int, int]:
        if self.max_n_samples is not None:
            used_samples = sum(sum(cont.N for cont in batch) for batch in batches)
            remaining_samples = max(0, self.max_n_samples - used_samples)
            samples_per_batch = remaining_samples // len(batches)
            extra_samples = remaining_samples % len(batches)
        else:
            samples_per_batch = None
            extra_samples = 0
        return samples_per_batch, extra_samples

    def _fit_batch_gp(self, batch: List[Container], integrand, n_samples: Optional[int], 
                      verbose: bool, grid_size: float, return_std: bool):
        if n_samples is None:
            n_samples = self.integral.n_samples

        try:
            kernel = self.integral.kernel
        except AttributeError:
            print('self.integral must have attributes `kernel`')
            return

        gp = self.integral.GPFit(**self.integral.gp_params)
        iGP = IterativeGPFitting(n_samples=n_samples, 
                                 n_splits=self.integral.n_splits, 
                                 max_redraw=self.integral.max_redraw, 
                                 performance_threshold=self.integral.threshold, 
                                 threshold_direction=self.integral.threshold_direction,
                                 gp=gp, fit_residuals=self.integral.fit_residuals)
        
        if verbose:
            total_n = np.sum([cont.N for cont in batch])
            print(f"Fitting a batch of containers with {total_n} data points"
                  f" and {len(batch)} containers")

        self._integrate_batch(iGP, kernel, batch, integrand, grid_size, return_std)

    def _integrate_batch(self, iGP, kernel, batch: List[Container], integrand, grid_size: float,
                         return_std: bool):
        try:
            gp_results = iGP.fit(integrand, batch, kernel, add_samples=True)
            gp = iGP.gp
        except Exception as e:
            print(f"GP fitting failed for batch: {e}")
            traceback.print_exc()
            return

        hyper_params = gp.hyper_params

        for container in batch:
            try:
                integral_result = kernel_integration(iGP, container, gp_results, return_std)
                self.integral_results[container] = integral_result
                container.hyper_params = hyper_params
            except Exception as e:
                print(f"Failed to process container {container}: {e}")
                traceback.print_exc()
                return

            neighbors = find_neighbors_grid(self._build_batches(batch, grid_size), container, grid_size)
            for neighbor in neighbors:
                neighbor.hyper_params = hyper_params