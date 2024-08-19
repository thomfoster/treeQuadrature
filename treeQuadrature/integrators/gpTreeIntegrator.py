from typing import List, Optional
import warnings, time, traceback
from queue import SimpleQueue
from inspect import signature
from traceback import print_exc
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..splits import Split
from .integrator import Integrator
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler, UniformSampler
from ..container import Container
from ..exampleProblems import Problem
from ..gaussianProcess import kernel_integration, IterativeGPFitting


default_sampler = UniformSampler()

class TreeNode:
    """
    A node in the tree 

    Attributes
    ----------
    container : Container
        the container represented by the node
    left : Container or None
        left child
    right : Container or None
        right child
    hyper_params : dict
        hyper-parameters for fitting GP
    """
    def __init__(self, container: Container):
        self.container = container
        self.left = None
        self.right = None
        self.hyper_params = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, container: Container):
        if self.root is None:
            self.root = TreeNode(container)
        else:
            self._insert(self.root, container)

    def _insert(self, node: TreeNode, container: Container):
        if node.left is None:
            node.left = TreeNode(container)
        elif node.right is None:
            node.right = TreeNode(container)
        else:
            self._insert(node.left, container)

    def get_leaf_nodes(self) -> List[TreeNode]:
        leaf_nodes = []
        self._get_leaf_nodes(self.root, leaf_nodes)
        return leaf_nodes

    def _get_leaf_nodes(self, node: TreeNode, leaf_nodes: List[TreeNode]):
        if node:
            if node.left is None and node.right is None:
                leaf_nodes.append(node)
            self._get_leaf_nodes(node.left, leaf_nodes)
            self._get_leaf_nodes(node.right, leaf_nodes)

    def get_leaf_containers(self) -> List[Container]:
        leaf_nodes = self.get_leaf_nodes()
        return [node.container for node in leaf_nodes]


def build_grid(leaf_nodes: List[TreeNode], grid_size: int) -> dict:
    grid = {}
    for node in leaf_nodes:
        mins = node.container.mins
        maxs = node.container.maxs
        grid_key = tuple(((mins + maxs) / 2 // grid_size).astype(int))
        if grid_key not in grid:
            grid[grid_key] = []
        grid[grid_key].append(node)
    return grid

def find_neighbors_grid(grid: dict, node: TreeNode, grid_size: int) -> List[TreeNode]:
    neighbors = []
    mins = node.container.mins
    maxs = node.container.maxs
    grid_key = tuple(((mins + maxs) / 2 // grid_size).astype(int))
    for offset in np.ndindex((3,) * len(grid_key)):
        neighbor_key = tuple(np.array(grid_key) + np.array(offset) - 1)
        if neighbor_key in grid:
            for neighbor in grid[neighbor_key]:
                if neighbor != node:
                    neighbors.append(neighbor)
    return neighbors


class GpTreeIntegrator(Integrator):
    def __init__(self, base_N: int, P: int, split: Split, integral: ContainerIntegral, 
                 grid_size: int, max_n_samples: Optional[int]=None,
                 sampler: Sampler=default_sampler):
        '''
        An integrator that allows communications between nodes
        for better GP tuning and fitting

        Attributes
        ----------
        base_N : int
            total number of initial samples
        P : int
            maximum number of samples in each container
        split : Split
            a method to split a container (for tree construction)
        integral : ContainerIntegral 
            a method to evaluate the integral of f on a container
            it must have return_hyper_params option
        sampler : Sampler
            a method for generating initial samples
            when problem does not have rvs method. 
            Default: UniformSampler
        max_n_samples : int
            if set, samples will be distributed 
            evenly across batches
            until max_n_samples samples are used
        
        Methods
        -------
        __call__(problem, return_N, return_all)
            solves the problem given

        Example
        -------
        >>> from treeQuadrature.integrators import SimpleIntegrator
        >>> from treeQuadrature.splits import MinSseSplit
        >>> from treeQuadrature.containerIntegration import RandomIntegral
        >>> from treeQuadrature.exampleProblems import SimpleGaussian
        >>> problem = SimpleGaussian(D=2)
        >>> 
        >>> minSseSplit = MinSseSplit()
        >>> randomIntegral = RandomIntegral()
        >>> integ = SimpleIntegrator(N=2_000, P=40, minSseSplit, randomIntegral)
        >>> estimate = integ(problem)
        >>> print("error of random integral =", 
        >>>      str(100 * np.abs(estimate - problem.answer) / problem.answer), "%")
        '''
        self.base_N = base_N
        self.split = split
        self.integral = integral
        self.sampler = sampler
        self.P = P
        self.max_n_samples = max_n_samples
        self.integral_results = {}
        self.grid_size = grid_size

    def construct_tree(self, root: Container, verbose: bool = False, max_iter: int = 1e4) -> BinaryTree:
        tree = BinaryTree()
        tree.insert(root)
        q = SimpleQueue()
        q.put(tree.root)

        start_time = time.time()
        iteration_count = 0

        while not q.empty() and iteration_count < max_iter:
            iteration_count += 1
            node = q.get()
            container = node.container

            if container.N <= self.P:
                continue
            else:
                children = self.split.split(container)
                node.left = TreeNode(children[0])
                node.right = TreeNode(children[1])
                q.put(node.left)
                q.put(node.right)

            if iteration_count % 100 == 0 and verbose:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.qsize()}, "
                      f"Elapsed time = {elapsed_time:.2f}s")

        total_time = time.time() - start_time
        if verbose:
            print(f"Total iterations: {iteration_count}")
            print(f"Total time taken: {total_time:.2f}s")

        if iteration_count == max_iter:
            warnings.warn(
                'maximum iterations reached, either '
                'increase max_iter or check split and samples', 
                RuntimeWarning)
                
        return tree

    def fit_gps(self, tree: BinaryTree, integrand, verbose: bool = False, 
            return_std: bool=False):
        leaf_nodes = tree.get_leaf_nodes()
        grid = build_grid(leaf_nodes, self.grid_size)
        
        batches = [nodes for _, nodes in grid.items()]

        if self.max_n_samples is not None:
            # Calculate remaining samples to distribute among batches
            used_samples = sum(node.container.N for node in leaf_nodes)
            remaining_samples = max(0, self.max_n_samples - used_samples)

            # Distribute remaining samples across batches
            samples_per_batch = remaining_samples // len(batches)
            extra_samples = remaining_samples % len(batches)
        else:
            samples_per_batch = None

        def fit_batch(batch, n_samples=None):
            containers = [node.container for node in batch]
            if n_samples is None:
                n_samples = self.integral.n_samples

            initial_N = sum([cont.N for cont in containers])

            # Check if hyperparameters are available and set them, otherwise use defaults
            try:
                kernel = self.integral.kernel
            except AttributeError:
                print('self.integral must have attributes `kernel`') 

            try: 
                gp = self.integral.GPFit(**self.integral.gp_params)
                iGP = IterativeGPFitting(n_samples=n_samples, 
                                        n_splits=self.integral.n_splits, 
                                        max_redraw=self.integral.max_redraw, 
                                        performance_threshold=self.integral.threshold, 
                                        threshold_direction=self.integral.threshold_direction,
                                        gp=gp, fit_residuals=self.integral.fit_residuals)
            except Exception:
                print("Failed to create GPFit instance")
                print_exc()
                return
            
            # Set kernel parameters using a representative batch
            representative_hyper_params = None
            for node in batch:
                if node.hyper_params is not None:
                    representative_hyper_params = node.hyper_params
                    break
            
            if representative_hyper_params is None:
                representative_hyper_params = kernel.get_params()
            kernel.set_params(**representative_hyper_params)

            if verbose:
                total_n = np.sum([cont.N for cont in containers])
                print(f"Fitting a batch of containers with {total_n} data points"
                      f" and {len(containers)} containers")

            try:
                gp_results = iGP.fit(integrand, containers, kernel, add_samples=True)
                gp = iGP.gp
            except Exception as e:
                print(f"GP fitting failed for batch: {e}")
                traceback.print_exc()
                return

            hyper_params = gp.hyper_params

            for node in batch:
                container = node.container
                try:
                    integral_result = kernel_integration(iGP, container, gp_results,
                                                         return_std)
                    self.integral_results[node] = integral_result
                    node.hyper_params = hyper_params
                except Exception as e:
                    print(f"Failed to process node {node}: {e}")
                    traceback.print_exc()
                    return

                # Pass hyper-parameters to neighbors
                neighbors = find_neighbors_grid(grid, node, self.grid_size)
                for neighbor in neighbors:
                    neighbor.hyper_params = hyper_params

            # check correct number of samples being added
            new_samples = sum([cont.N for cont in containers]) - initial_N

        # Parallel processing of batches
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fit_batch, batch, n_samples=samples_per_batch + (1 if i < extra_samples else 0))
                for i, batch in enumerate(batches)
            ]
            for future in futures:
                future.result()

    def __call__(self, problem: Problem, return_N: bool = False, 
                 return_containers: bool = False, return_std: bool = False, 
                 verbose: bool = False, 
                 *args, **kwargs) -> dict:
        if verbose: 
            print('drawing initial samples')

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

        if verbose:
            print('constructing tree')
            if 'verbose' in signature(self.construct_tree).parameters:
                tree = self.construct_tree(root, verbose=True, *args, **kwargs)
            else:
                tree = self.construct_tree(root, *args, **kwargs)
        else: 
            tree = self.construct_tree(root, *args, **kwargs)

        if verbose:
            print('fitting GP to containers and passing hyper-parameters')
        self.fit_gps(tree, problem.integrand, verbose, return_std)

        leaf_nodes = tree.get_leaf_nodes()

        if return_std:
            if hasattr(self.integral, 'return_std'):
                contributions = [self.integral_results[node]['integral'] for node in leaf_nodes]
                stds = [self.integral_results[node]['std'] for node in leaf_nodes]
            else:
                warnings.warn(
                    f'{str(self.integral)} does not '
                     'have parameter return_std, will be ignored', 
                     UserWarning)
                return_std = False
                contributions = [self.integral_results[node]['integral'] for node in leaf_nodes]
        else: 
            # Collect contributions from integral results
            contributions = []
            missing_nodes = []
            for node in leaf_nodes:
                try:
                    contributions.append(self.integral_results[node]['integral'])
                except KeyError:
                    missing_nodes.append(node)

            if missing_nodes:
                raise RuntimeError(
                    f"Missing integral results for {len(missing_nodes)} of {len(leaf_nodes)} containers."
                    )

        G = np.sum(contributions)
        N = sum([node.container.N for node in leaf_nodes])

        return_values = {'estimate' : G}
        if return_N:
            return_values['n_evals'] = N
        if return_containers:
            return_values['containers'] = [node.container for node in leaf_nodes]
            return_values['contributions'] = contributions
        if return_std:
            return_values['stds'] = stds

        return return_values
        

    