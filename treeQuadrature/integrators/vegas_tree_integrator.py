import vegas
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature

from ..container import Container
from .tree_integrator import TreeIntegrator
from .distributed_tree_integrator import DistributedTreeIntegrator
from ..trees import Tree
from ..example_problems import Problem
from ..container_integrators import ContainerIntegral


class TransformedProblem(Problem):
    def __init__(self, D: int, map: vegas.AdaptiveMap,
                 original_integrand: callable):
        # y-space is the unit hypercube [0, 1]^D
        super().__init__(D, lows=0, highs=1)
        self.map = map
        self.original_integrand = original_integrand

    def integrand(self, xs) -> Tuple[np.ndarray, np.ndarray]:
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
        return "Transformed Problem under Vegas Map"


def generate_uniform_grid(n_samples):
    # Generate a uniform grid of samples in [0, 1]^2
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    X, Y = np.meshgrid(x, y)
    samples = np.vstack([X.ravel(), Y.ravel()]).T
    return samples


def plot_transformed_samples(
    vegas_integrator,
    n_samples: int = 20,
    font_size: int = 20,
    tick_size: int = 18,
    title: str = None,
    file_path: Optional[str] = None,
    plot_original: bool = False,
):
    """
    Plot the transformed samples in the transformed space.

    Parameters
    ----------
    vegas_integrator : vegas.Integrator
        The traind Vegas integrator object
    n_samples : int, optional
        The number of samples to generate in [0, 1]^2
        in each dimension. \n
        Default: 20
    font_size : int, optional
        The font size of the plot. \n
        Default: 20
    tick_size : int, optional
        The tick size of the plot. \n
        Default: 18
    title : str, optional
        The title of the plot. \n
        if None, no title is shown.
    file_path : str, optional
        The path to save the plot. \n
        If None, the plot is shown.
    plot_original : bool, optional
        Whether to plot the original samples in [0, 1]^2
    """

    samples = generate_uniform_grid(n_samples)

    # Ensure samples are C-contiguous
    samples = np.ascontiguousarray(samples)

    # Transform the uniform samples using vegas_integrator.map.map
    X_mapped = np.empty_like(samples)
    jac = np.empty(samples.shape[0])
    vegas_integrator.map.map(samples, X_mapped, jac)

    if plot_original:
        # Plot original samples and transformed samples side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original samples in [0, 1]^2
        ax1.scatter(
            samples[:, 0], samples[:, 1],
            c="blue", marker="o", alpha=0.5)
        ax1.grid(True)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.xaxis.set_tick_params(labelsize=tick_size)
        ax1.yaxis.set_tick_params(labelsize=tick_size)

        # Plot transformed samples in [0, 1]^2
        ax2.scatter(
            X_mapped[:, 0], X_mapped[:, 1],
            c="blue", marker="o", alpha=0.5)
        ax2.grid(True)
        ax2.xaxis.set_tick_params(labelsize=tick_size)
        ax2.yaxis.set_tick_params(labelsize=tick_size)

    else:
        # Plot only the transformed samples
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_mapped[:, 0], X_mapped[:, 1],
            c="blue", marker="o", alpha=0.5)
        plt.title(title, fontsize=font_size + 3)
        plt.grid(True)

    # Set overall title
    if title:
        plt.suptitle(title, fontsize=font_size + 3)

    # Save or show plot
    if file_path is not None:
        plt.savefig(file_path, dpi=400)
        print(f"figure saved to {file_path}")
    else:
        plt.show()


class VegasTreeIntegrator(TreeIntegrator):
    def __init__(
        self,
        vegas_N: int,
        tree: Optional[Tree] = None,
        integral: Optional[ContainerIntegral] = None,
        vegas_iter: int = 10,
        max_N: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Integrator that combines the Vegas algorithm with
        the tree quadrature algorithm.

        Parameters
        ----------
        vegas_N : int
            The number of samples for building Vegas map
        tree : Tree, optional
            The tree structure to use. \n
            If None, a default SimpleTree is used
        integral : ContainerIntegral, optional
            The container integrator to use. \n
            If None, a default RandomIntegral is used.
        vegas_iter : int, optional
            The number of iterations for the Vegas algorithm.
            Default: 10
        max_N : int, optional
            The maximum number of evaluations allowed.
            If given, DistributedTreeIntegrator will be used.
        **kwargs : Any
            Additional keyword arguments to initialise the
            DistributedTreeIntegrator when specifying max_N
        """

        # base_N used for drawing samples not used here, set to 0
        # as Vegas samples are used
        super().__init__(0, tree, integral)
        self.vegas_iter = vegas_iter
        self.vegas_N = vegas_N
        if max_N:
            if max_N <= vegas_N:
                raise ValueError("max_N must be greater than vegas_N")
            self.limit_samples = True
            dist_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in signature(DistributedTreeIntegrator).parameters
            }
            self.distributed = DistributedTreeIntegrator(
                0, max_N, **dist_kwargs)
        else:
            self.limit_samples = False

    def __call__(
        self,
        problem: Problem,
        return_N: bool = False,
        return_std: bool = False,
        return_containers: bool = False,
        return_raw: bool = False,
        verbose: bool = False,
        plot_vegas: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
            self.vegas_N, problem, plot_vegas, verbose, **kwargs
        )

        # Construct tree on the transformed space
        root = Container(X_transformed, y_transformed, mins=0, maxs=1)
        finished_containers = self._construct_tree(
            root, problem, verbose, **kwargs)

        # Integrate the transformed problem
        problem_transformed = TransformedProblem(
            problem.D, vegas_integrator.map, problem.integrand
        )

        if self.limit_samples:
            results, modified_containers = (
                self.distributed.integrate_containers(
                    finished_containers, problem_transformed,
                    compute_std, verbose, **kwargs
                )
            )
        else:
            results, modified_containers = self.integrate_containers(
                finished_containers, problem_transformed,
                compute_std, verbose, **kwargs
            )

        return self._compile_results(
            results,
            modified_containers,
            compute_std,
            return_N,
            return_containers,
            return_raw,
        )

    def _build_vegas_map(
        self, N, problem: Problem, plot_vegas: bool, verbose, **kwargs
    ):
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
            jac = np.empty(x.shape[0])  # Jacobin
            vegas_integrator.map.invmap(x, y, jac)
            # Store the y-space values and corresponding integrand values
            y_list.append(y)
            jac_list.append(jac)
            integrand_values_list.append(f_vals * jac)
            return f_vals

        vegas_integrator = vegas.Integrator(domain_bounds)
        vegas_n = int(N // self.vegas_iter)

        if verbose:
            print(
                f"Building Vegas map with {self.vegas_iter} iterations "
                f"and {vegas_n} samples per iteration"
            )

        vegas_integrator(batch_integrand, nitn=self.vegas_iter, neval=vegas_n)

        if plot_vegas:
            plot_parameters = signature(plot_transformed_samples).parameters
            applicable_kwargs = {
                k: v for k, v in kwargs.items() if k in plot_parameters
            }

            plot_transformed_samples(vegas_integrator, **applicable_kwargs)

        # Step 2: Transform the space using Vegas map
        X_transformed = np.vstack(y_list)
        y_transformed = np.vstack(integrand_values_list)

        return vegas_integrator, X_transformed, y_transformed
