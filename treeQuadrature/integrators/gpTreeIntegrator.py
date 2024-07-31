from typing import List
import warnings, time, traceback
from queue import SimpleQueue
from inspect import signature
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..splits import Split
from .integrator import Integrator
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler, UniformSampler
from ..container import Container
from ..exampleProblems import Problem
from ..gaussianProcess import kernel_integration


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
                 grid_size: int,
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
        if 'return_hyper_params' not in signature(
            integral.containerIntegral).parameters:
            raise AssertionError(
                'integral.containerIntegral must have'
                ' return_hyper_params option'
                ' and return the hyper-parameters as a dictionary'
                ' following integral estimate'
                )
        self.sampler = sampler
        self.P = P
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

        def fit_batch(batch):
            containers = [node.container for node in batch]

            # Check if hyperparameters are available and set them, otherwise use defaults
            try:
                kernel = self.integral.kernel
                iGP = self.integral.iGP
                threshold = self.integral.threshold
                threshold_direction = self.integral.threshold_direction
            except AttributeError:
                print('containerIntegral must have attributes `kernel`, '
                      '`iGP`, `threshold`, `threshold_direction`') 
                
            representative_hyper_params = None

            # Select a representative hyperparameter set
            # TODO - design a more intelligent selection? 
            for node in batch:
                if node.hyper_params is not None:
                    representative_hyper_params = node.hyper_params
                    break
            
            if representative_hyper_params is None:
                # If no hyperparameters are set, use the default kernel parameters
                representative_hyper_params = kernel.get_params()

            # Set the kernel parameters with the selected hyperparameters
            kernel.set_params(**representative_hyper_params)

            if verbose:
                total_n = np.sum([cont.N for cont in containers])
                print(f"Fitting a batch of containers with {total_n} data points")

            try:
                performance = iGP.fit(integrand, containers, kernel, add_samples=True)
                gp = iGP.gp
            except Exception as e:
                print(f"GP fitting failed for batch: {e}")
                traceback.print_exc()
                return

            hyper_params = gp.hyper_params

            for node in batch:
                container = node.container
                try:
                    integral_result = kernel_integration(gp, container, 
                                                                    return_std, performance, 
                                                                    threshold, threshold_direction)
                    self.integral_results[node] = integral_result
                    node.hyper_params = hyper_params
                except Exception as e:
                    raise Exception(f"Failed to process node {node}: {e}")

                # Pass hyper-parameters to neighbors
                neighbors = find_neighbors_grid(grid, node, self.grid_size)
                for neighbor in neighbors:
                    neighbor.hyper_params = hyper_params

        # parallel processing
        with ThreadPoolExecutor() as executor:
            executor.map(fit_batch, batches)

    def __call__(self, problem: Problem, return_N: bool = False, 
                 return_containers: bool = False, return_std: bool = False, 
                 verbose: bool = False, *args, **kwargs) -> dict:
        if verbose: 
            print('drawing initial samples')

        if hasattr(problem, 'rvs'):
            X = problem.rvs(self.base_N)
        else:
            X = self.sampler.rvs(self.base_N, problem)
        y = problem.integrand(X)
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
                contributions = [self.integral_results[node][0] for node in leaf_nodes]
                stds = [self.integral_results[node][1] for node in leaf_nodes]
            else:
                warnings.warn(
                    f'{str(self.integral)} does not '
                     'have parameter return_std, will be ignored', 
                     UserWarning)
                return_std = False
                contributions = [self.integral_results[node] for node in leaf_nodes]
        else: 
            # Collect contributions from integral results
            contributions = []
            missing_nodes = []
            for node in leaf_nodes:
                try:
                    contributions.append(self.integral_results[node])
                except KeyError:
                    missing_nodes.append(node)

            if missing_nodes:
                raise RuntimeError(
                    f"Missing integral results for {len(missing_nodes)} of {len(leaf_nodes)} containers."
                    )
            contributions = [self.integral_results[node] for node in leaf_nodes]

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
        

    