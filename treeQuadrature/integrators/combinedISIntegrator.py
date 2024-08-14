from typing import List
import warnings, time
from queue import SimpleQueue
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from ..splits import Split
from .integrator import Integrator
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler, UniformSampler
from ..container import Container, ArrayList
from ..exampleProblems import Problem

default_sampler = UniformSampler()

class CombinedISIntegrator(Integrator):
    def __init__(self, base_N: int, P: int, split: Split, integral: ContainerIntegral, 
                 sampler: Sampler=default_sampler):
        '''
        An integrator that uses Iterative Importance Sampling to estimate the integral

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
        '''
        self.base_N = base_N
        self.split = split
        self.integral = integral
        self.sampler = sampler
        self.P = P
        self.integral_results = {}

    def construct_tree(self, containers: List[Container],
                       verbose: bool=False, max_iter: int=1e5):
        """
        Construct a tree of containers.

        Parameters
        ----------
        root : Container
            The root container.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        max_iter : int, optional
            Maximum number of iterations, 
            by default 1e4.

        Returns
        -------
        List[Container]
            A list of finished containers.
        """
        # Construct tree
        finished_containers = []
        q = SimpleQueue()
        for c in containers:
            q.put(c)

        start_time = time.time()
        iteration_count = 0

        while not q.empty() and iteration_count < max_iter:
            iteration_count += 1
            c = q.get()

            if c.N <= self.P:
                finished_containers.append(c)
            else:
                children = self.split.split(c)
                if len(children) == 1:
                    finished_containers.append(c)
                else:
                    for child in children:
                        q.put(child)
            
            if iteration_count % 100 == 0 and verbose:  # Log every 100 iterations
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration_count}: Queue size = {q.qsize()}, "
                    f"number of containers = {len(finished_containers)}, "
                    f"Elapsed time = {elapsed_time:.2f}s")
                
        total_time = time.time() - start_time
        if verbose:
            print(f"Total finished containers: {len(finished_containers)}")
            print(f"Total iterations: {iteration_count}")
            print(f"Total time taken: {total_time:.2f}s")
        
        if iteration_count == max_iter:
            warnings.warn(
                'maximum iterations reached, either '
                'incresae max_iter or check split and samples', 
                RuntimeWarning)
                
        return finished_containers


    def __call__(self, problem: Problem, N_iter: int = 5, N_initMC: int = 30, 
                 N_eval: int = 10_000,return_N: bool = False, return_containers: bool = False, 
                 return_std: bool = False, verbose: bool = False, *args, **kwargs) -> dict:

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
        containers = self.construct_tree([root], *args, **kwargs)

        # clear all container X and y values
        # generate N_initial new MC samples in each container
        for c in containers:
            c._X = ArrayList(D=problem.D)
            c._y = ArrayList(D=1)
            new_X = np.random.uniform(low=c.mins, high=c.maxs, size=(N_initMC, problem.D))
            new_y = problem.integrand(new_X)
            c.add(new_X, new_y)

        I_hats = np.empty(N_iter+1)

        mean = np.array([0.]*problem.D)
        covar = np.diag([1/200]*problem.D)
        dist = mvn(mean=mean, cov=covar)

        # calculate estimate of integral
        J_hat = np.sum([np.mean(c.y)*c.volume for c in containers])

        # calculate probabilities to sample containers from
        cont_vols = np.array([np.sqrt(np.var(c.y))*c.volume for c in containers])
        tot_vol = sum(cont_vols)
        probabilities = cont_vols/tot_vol

        # sample new x points from IS bias distribution
        sample_containers = np.random.choice(containers, p=probabilities, size=self.base_N)
        densities = (np.array([np.sqrt(np.var(c.y)) for c in sample_containers]))/tot_vol       
        xs = np.array([np.random.uniform(low=c.mins, high=c.maxs, size=problem.D) for c in sample_containers])
        ys = problem.integrand(xs).reshape(-1)
        means = np.array([np.mean(c.y) for c in sample_containers])

        # construct estimates
        I_hats[0] = J_hat + sum((ys-means)/densities)/N_eval

        print(J_hat)
        print('I0 = ' + str(I_hats[0]))

        # expand tree
        for c in containers:
            c.add(xs[sample_containers == c], ys[sample_containers == c])
        containers = self.construct_tree(containers)

        for c in containers:
            new_X = np.random.uniform(low=c.mins, high=c.maxs, size=(N_initMC, problem.D))
            new_y = problem.integrand(new_X)
            c.add(new_X, new_y)

        for i in range(N_iter):
            # calculate probabilities to sample containers from
            cont_vols = np.array([np.sqrt(np.mean(c.y**2))*c.volume for c in containers])
            tot_vol = sum(cont_vols)
            probabilities = cont_vols/tot_vol

            # sample new x points from IS distribution
            sample_containers = np.random.choice(containers, p=probabilities, size=N_eval)
            densities = np.array([np.sqrt(np.mean(c.y**2)) for c in sample_containers])/tot_vol            
            xs = np.array([np.random.uniform(low=c.mins, high=c.maxs, size=problem.D) for c in sample_containers])
            ys = problem.integrand(xs).reshape(-1)

            # construct estimates
            I_hats[i+1] = sum(ys/densities)/N_eval

            # update containers
            for c in containers:
                c.add(xs[sample_containers == c], ys[sample_containers == c])

            print('I' + str(i+1) + ' = ' + str(I_hats[i+1]))

            # plt.hist(ys/densities)
            # plt.show()

        # plt.scatter([len(c.y) for c in containers],
        #             [np.mean(c.y)*c.volume - dist.cdf(c.maxs,lower_limit=c.mins) for c in containers])
        # plt.show()
        
        I = sum(I_hats)/(N_iter+1)
        print('final results:  I = ' + str(I))
        print(sum([c.N for c in containers]))








        
        
        

