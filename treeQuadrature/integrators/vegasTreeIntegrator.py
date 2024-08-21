import vegas
from typing import Dict, Any, List
import numpy as np
from inspect import signature
import matplotlib.pyplot as plt

from ..container import Container
from .treeIntegrator import TreeIntegrator
from .simpleIntegrator import SimpleIntegrator
from ..exampleProblems import Problem
from ..samplers import Sampler
from ..splits import Split
from ..containerIntegration import ContainerIntegral
from ..visualisation import plotIntegrand

class TransformedProblem(Problem):
    def __init__(self, D: int, map: vegas.AdaptiveMap, 
                    original_integrand: callable):
        # y-space is the unit hypercube [0, 1]^D
        super().__init__(D, lows=0, highs=1)
        self.map = map
        self.original_integrand = original_integrand

    def integrand(self, xs) -> np.ndarray:
        xs = self.handle_input(xs)
    
        # Transform Y to X using the VEGAS map
        X_mapped = np.empty_like(xs)
        jac = np.empty(xs.shape[0])
        self.map.map(xs, X_mapped, jac)  # Transform samples
        
        f_vals = self.original_integrand(X_mapped)
        
        # Ensure the output shape is (N, 1)
        if f_vals.ndim == 1:
            f_vals = f_vals[:, np.newaxis]
        
        # Multiply by the Jacobian and return
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

def plot_transformed_samples(vegas_integrator, n_samples):
    samples = generate_uniform_grid(n_samples)

    # Ensure samples are C-contiguous
    samples = np.ascontiguousarray(samples)

    # Transform the uniform samples using vegas_integrator.map.map
    X_mapped = np.empty_like(samples)
    jac = np.empty(samples.shape[0])
    vegas_integrator.map.map(samples, X_mapped, jac)

    # Plot the transformed samples
    plt.figure(figsize=(8, 6))
    plt.scatter(X_mapped[:, 0], X_mapped[:, 1], c='blue', marker='o', alpha=0.5)
    plt.xlabel('X dimension 1')
    plt.ylabel('X dimension 2')
    plt.title('Transformed Samples in X-space')
    plt.grid(True)
    plt.show()

class VegasTreeIntegrator(TreeIntegrator):
    def __init__(self, base_N: int, P: int, split: Split,
                 integral: ContainerIntegral, 
                 vegas_iter: int=10):
        super().__init__(split, integral, base_N)
        self.P = P
        self.vegas_iter = vegas_iter

    def construct_tree(self, root: Container, *args, **kwargs) -> List[Container]:
        # borrow the construct_tree method
        # TODO - tree construction might have to be a separate abstract class 
        temporary = SimpleIntegrator(self.base_N, self.P, 
                                     self.split, self.integral, self.sampler)
        return temporary.construct_tree(root, *args, **kwargs)

    def __call__(self, problem: Problem, return_N: bool, return_std: bool=False,
                 return_containers: bool=False,
                 **kwargs: Any) -> Dict[str, Any]:
        
        method = getattr(self.integral, 'containerIntegral', None)
        if method:
            has_return_std =  'return_std' in signature(method).parameters
        else:
            raise TypeError("self.integral must have 'containerIntegral' method")
        compute_std = return_std and has_return_std

        # Step 1: Use half of the evaluations to create a VEGAS map
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
        vegas_n = int(self.base_N // self.vegas_iter)
        vegas_integrator(batch_integrand, nitn=self.vegas_iter, 
                         neval=vegas_n)
        
        plot_transformed_samples(vegas_integrator, 100)

        # Step 2: Transform the space using Vegas map
        X_transformed = np.vstack(y_list)
        y_transformed = np.vstack(integrand_values_list)

        # Step 3: Integrate in the transformed space using Tree Quadrature
        root = Container(X_transformed, y_transformed, 
                         mins=0, maxs=1)
        finished_containers = self.construct_tree(root, *kwargs.values())
            
        problem_transformed = TransformedProblem(problem.D, 
                                                 vegas_integrator.map, 
                                                 problem.integrand)
        
        plotIntegrand(problem_transformed.integrand, problem.D, 
                      xlim = [0, 1], ylim=[0, 1], levels=30)

        results, modified_containers = self.integrate_containers(finished_containers, 
                                                                 problem_transformed, 
                                                                 compute_std)
        contributions = [result['integral'] for result in results]
        stds = [result['std'] for result in results] if compute_std else None

        estimate = sum(contributions)

        n_evals = [cont.N for cont in modified_containers]

        # Prepare the result dictionary
        return_values = {'estimate' : estimate}
        if return_N:
            return_values['n_evals'] = n_evals
        if return_containers:
            return_values['containers'] = modified_containers
            return_values['contributions'] = contributions
        if compute_std and stds is not None:
            return_values['stds'] = stds

        return return_values