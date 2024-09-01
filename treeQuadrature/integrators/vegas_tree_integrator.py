import vegas
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature

from ..container import Container
from .tree_integrator import TreeIntegrator
from ..trees import Tree
from ..example_problems import Problem
from ..samplers import Sampler, UniformSampler
from ..container_integrators import ContainerIntegral
from ..visualisation import plot_integrand

class TransformedProblem(Problem):
    def __init__(self, D: int, map: vegas.AdaptiveMap, 
                    original_integrand: callable):
        # y-space is the unit hypercube [0, 1]^D
        super().__init__(D, lows=0, highs=1)
        self.map = map
        self.original_integrand = original_integrand

    def integrand(self, xs) -> Tuple[
        np.ndarray, np.ndarray]:
        xs = self.handle_input(xs)

        # Transform Y to X using the VEGAS map
        X_mapped = np.empty_like(xs)
        jac = np.empty(xs.shape[0])
        self.map.map(xs, X_mapped, jac)  # Transform samples

        f_vals = self.original_integrand(X_mapped)

        # Ensure the output shape is (N, 1)
        if f_vals.ndim == 1:
            f_vals = f_vals[:, np.newaxis]

        # return the integrand values and the jacobian
        return f_vals * jac[:, np.newaxis]

    def __str__(self) -> str:
        return 'Transformed Problem under Vegas Map'


def generate_uniform_grid(n_samples):
    # Generate a uniform grid of samples in [0, 1]^2
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    X, Y = np.meshgrid(x, y)
    samples = np.vstack([X.ravel(), Y.ravel()]).T
    return samples


def plot_transformed_samples(vegas_integrator, n_samples: int=100,
                             font_size: int=15, title: str=None,
                             file_path: Optional[str]=None):
    samples = generate_uniform_grid(n_samples)

    # Ensure samples are C-contiguous
    samples = np.ascontiguousarray(samples)

    # Transform the uniform samples using vegas_integrator.map.map
    X_mapped = np.empty_like(samples)
    jac = np.empty(samples.shape[0])
    vegas_integrator.map.map(samples, X_mapped, jac)

    # Plot the transformed samples
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_mapped[:, 0], X_mapped[:, 1],
        c='blue', marker='o', alpha=0.5)
    plt.title(title, fontsize=font_size+3)
    plt.grid(True)
    
    if file_path is not None:
        plt.savefig(file_path)
        print(f"figure saved to {file_path}")
    else:
        plt.show()


class VegasTreeIntegrator(TreeIntegrator):
    def __init__(self, tree_N: int, vegas_N: int,
                 tree: Optional[Tree]=None,
                 integral: Optional[ContainerIntegral]=None,
                 sampler: Optional[Sampler]=None,
                 vegas_iter: int=10
        ):
        """
        Integrator that combines the Vegas algorithm with
        the tree quadrature algorithm.

        Parameters
        ----------
        tree_N : int
            number of samples for building the tree
        vegas_N : int
            The number of samples for building Vegas map
        tree : Tree, optional
            The tree structure to use. \n
            If None, a default SimpleTree is used
        integral : ContainerIntegral, optional
            The container integrator to use. \n
            If None, a default RandomIntegral is used. 
        sampler : Sampler, optional
            The sampler to use. \n
            If None, a default UniformSampler is used. 
        vegas_iter : int, optional
            The number of iterations for the Vegas algorithm. 
        """

        sampler = sampler if (
            sampler is not None) else UniformSampler()
        
        super().__init__(tree_N, tree, integral, sampler)
        self.vegas_iter = vegas_iter
        self.vegas_N = vegas_N

    def __call__(self, problem: Problem,
                 return_N: bool=False,
                 return_std: bool=False,
                 return_containers: bool=False,
                 return_raw: bool=False,
                 verbose: bool=False,
                 plot_vegas: bool=False,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : Problem
            The problem to integrate
        return_N : bool, optional
            Whether to return the total number of evaluations
        return_std : bool, optional
            Whether to return the standard deviation of the estimate
        return_containers : bool, optional
            Whether to return the containers
        return_raw : bool, optional
            Whether to return the raw results
        plot_vegas : bool, optional
            Whether to plot the Vegas adaptive map
        verbose : bool, optional
            Whether to print additional information
        **kwargs : Any
            Additional keyword arguments to pass to
            the tree construction or container integrator
        """

        compute_std = self._check_return_std(return_std)

        # Use half of the evaluations to create a VEGAS map
        vegas_integrator, X_transformed, y_transformed = self._build_vegas_map(
            self.vegas_N, problem, plot_vegas, **kwargs)

        # Step 3: Construct tree on the transformed space
        root = Container(X_transformed, y_transformed, 
                         mins=0, maxs=1)
        finished_containers = self._construct_tree(
            root, problem, verbose, **kwargs)

        # Step 4: Integrate the transformed problem
        problem_transformed = TransformedProblem(
            problem.D, vegas_integrator.map, problem.integrand)

        results, modified_containers = self.integrate_containers(
            finished_containers, problem_transformed, compute_std,
            **kwargs)

        return self._compile_results(
            results, modified_containers, compute_std, return_N,
            return_containers, return_raw)

    def _build_vegas_map(self, N,
                         problem: Problem,
                         plot_vegas: bool, **kwargs):
        domain_bounds = []
        for i in range(problem.D): 
            domain_bounds.append([problem.lows[i], problem.highs[i]])

        y_list = []
        jac_list = []
        integrand_values_list = []

        def batch_integrand(x):
            x = np.atleast_2d(x)
            # Evaluate the function at these points
            f_vals = problem.integrand(x)
            # Transform x to y using the VEGAS map
            y = np.empty_like(x)
            jac = np.empty(x.shape[0]) # Jacobin
            vegas_integrator.map.invmap(x, y, jac)
            # Store the y-space values and corresponding integrand values
            y_list.append(y)
            jac_list.append(jac)
            integrand_values_list.append(f_vals * jac)
            return f_vals

        vegas_integrator = vegas.Integrator(domain_bounds)
        vegas_n = int(N // self.vegas_iter)
        vegas_integrator(batch_integrand, nitn=self.vegas_iter, 
                         neval=vegas_n)

        if plot_vegas:
            plot_parameters = signature(
                plot_transformed_samples).parameters
            applicable_kwargs = {
                k: v for k, v in kwargs.items()
                if k in plot_parameters
            }

            plot_transformed_samples(
                vegas_integrator, **applicable_kwargs)

        # Step 2: Transform the space using Vegas map
        X_transformed = np.vstack(y_list)
        y_transformed = np.vstack(integrand_values_list)

        return vegas_integrator, X_transformed, y_transformed