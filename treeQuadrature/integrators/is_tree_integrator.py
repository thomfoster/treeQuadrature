from .tree_integrator import TreeIntegrator
from ..container import Container
from ..example_problems import Problem
from ..utils import ResultDict

from typing import List
import numpy as np


class ISTreeIntegrator(TreeIntegrator):
    """
    Use importance sampling to evaluate the integral
    """
    def __init__(self, base_N, tree = None,
                 sampler = None, parallel = True,
                 N_init: int=10,
                 N_eval: int=20_000,
                 *args, **kwargs):
        """
        Initialise the ISTreeIntegrator.

        Parameters
        ----------
        base_N : int
            The number of initial samples to draw.
        tree : Tree, Optional
            The tree to be used for the integration process.
            Defaults to SimpleTree.
        sampler : Sampler, Optional
            The sampler to be used to draw initial samples.
            If None, problem.rvs will be used.
        parallel : bool, Optional
            TODO - parallelisation not yet implemented
            Whether to use parallel computing.
            Defaults to True.
        N_init : int, Optional
            Number of initial samples used to determine
            container densities.
            Default: 10
        *args, **kwargs : Any
            Additional arguments to be passed to the tree construction method.
        """
        super().__init__(base_N, tree, None, sampler, parallel,
                         *args, **kwargs)
        self.N_init = N_init
        self.N_eval = N_eval
    
    def __call__(self, problem:Problem, return_N = False,
                 return_containers = False, return_std = False,
                 verbose = False,
                 *args, **kwargs):
        X, y = self._draw_initial_samples(problem, verbose)
        root = self._construct_root_container(X, y, problem, verbose)

        containers = self._construct_tree(
            root, problem, verbose, *args, **kwargs)
        
        result = self.integrate_containers(
            containers, problem, return_std, **kwargs
        )

        return self._compile_results(result, containers,
                                     return_N, return_std, return_containers)
    
    def _compile_results(self, result:dict, containers:List[Container],
                         return_N, return_std, return_containers):
        compiled_result = ResultDict(estimate=result['estimate'])
        if return_N:
            # total number of evaluations
            N = sum([cont.N for cont in containers])
            compiled_result['n_evals'] = N
        if return_std:
            compiled_result['std'] = result['std']
        if return_containers:
            compiled_result['containers'] = containers

        return compiled_result

    def integrate_containers(self, containers:List[Container], problem:Problem,
                             compute_std:bool = False,
                             **kwargs):
        # generate N_init new uniform samples in each container
        cont_ss = [] # record square root of sum of squared
        for c in containers:
            init_X = c.rvs(self.N_init)
            init_y = problem.integrand(init_X).reshape(-1)
            cont_ss.append(np.sqrt(np.mean(init_y**2)))

            c.add(init_X, init_y)
        
        # collect volumes
        cont_vols = np.array([c.volume for c in containers])
        # container densities (weights)
        weights = cont_ss * cont_vols
        probabilities = weights / sum(weights)

        cont_indices = np.random.choice(list(range(len(containers))),
                                p=probabilities,
                                size=self.N_eval)
        
        # Obtain the corresponding samples
        total_weighted_sum = 0
        weighted_squares_sum = 0

        for container_idx in np.unique(cont_indices):
            n_samples = np.sum(cont_indices == container_idx)
            xs = containers[container_idx].rvs(n_samples)
            ys = problem.integrand(xs).reshape(-1)

            containers[container_idx].add(xs, ys)

            # normalise the integrand value by sum of squares
            weighted_contributions = ys / probabilities[container_idx] * \
                cont_vols[container_idx]   # density of uniform sampling
            total_weighted_sum += np.sum(weighted_contributions)
            weighted_squares_sum += np.sum(weighted_contributions ** 2)

            
        estimate = total_weighted_sum / self.N_eval
        result = {'estimate' : estimate}
        if compute_std:
            # second moment
            M2_hat = weighted_squares_sum / self.N_eval
            variance_hat = (M2_hat - estimate**2) / self.N_eval
            result['std'] = np.sqrt(variance_hat)

        return result