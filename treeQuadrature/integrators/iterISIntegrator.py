from typing import List
import warnings, time
from queue import SimpleQueue
from inspect import signature
import numpy as np
import matplotlib.pyplot as plt


from ..splits import Split
from .integrator import Integrator
from ..containerIntegration import ContainerIntegral
from ..samplers import Sampler, UniformSampler
from ..container import Container, ArrayList
from ..exampleProblems import Problem

default_sampler = UniformSampler()

class IterISIntegrator(Integrator):
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

    def smooth(self, problem, mids, d):
        max_range = max(problem.highs-problem.lows)
        MSSE = np.empty(10)
        hs = np.linspace(max_range/100, max_range/10, 10)
        for iteration in range(3):
            for i in range(10):
                rows = [[np.exp(-np.sum((x-y)**2)/hs[i]**2) for x in mids] for y in mids]
                S = np.matrix([row/sum(row) for row in rows])
                smoothed = np.array(np.matmul(S,d)).reshape(-1)
                diags = np.diag(S)
                if np.isin(1., diags):
                    MSSE[i] = np.inf
                else:
                    MSSE[i] = sum(((d-smoothed)/(1-diags))**2)

            if np.argmin(MSSE) == 0:
                hs = np.linspace(hs[1]/10,hs[1],10)
            if np.argmin(MSSE) == 9:
                hs = np.linspace(hs[9],10*hs[9],10)
            else:
                hs = np.linspace(hs[np.argmin(MSSE)-1], hs[np.argmin(MSSE)+1], 10)

            if iteration == 3 and (np.argmin(MSSE) == 0 or np.argmin(MSSE) == 9):
                warnings.warn('Smoothing fitting may have failed')

        hmin = hs[np.argmin(MSSE)]
        plt.plot(hs, MSSE)
        plt.show()


        rows = [[np.exp(-np.sum((x-y)**2)/hmin**2) for x in mids] for y in mids]
        S = np.matrix([row/sum(row) for row in rows])
        return np.array(np.matmul(S,d)).reshape(-1)

        



    def __call__(self, problem: Problem, N_iter: int = 7, N_initMC: int = 10, 
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

        I_hats = np.empty(N_iter)
        Sig2_Hats = np.empty(N_iter)

        for i in range(N_iter):
            # calculate probabilities to sample containers from
            cont_means = np.array([np.sqrt(np.mean(c.y**2)) for c in containers])
            mids = [c.midpoint for c in containers]
            vols = np.array([c.volume for c in containers])

            # cont_means = self.smooth(problem, mids, cont_means)
            d = cont_means
            d = d*vols 
            d = ((d-1)/np.log(d))**1
            probabilities = d/sum(d)
            

            # sample new x points from IS distribution
            samp = np.random.choice([*range(len(containers))], p=probabilities, size=N_eval)
            densities = (np.array([d[s]/containers[s].volume for s in samp]))/sum(d)    
            xs = np.array([np.random.uniform(low=containers[s].mins, high=containers[s].maxs, size=problem.D) for s in samp])
            ys = problem.integrand(xs).reshape(-1)

            # construct estimates
            I_hats[i] = sum(ys/densities)/N_eval
            Sig2_Hats[i] = (sum(ys**2/densities**2)/N_eval - I_hats[i]**2)/N_eval

            
            # plt.scatter([np.linalg.norm(c.midpoint) for c in containers], problem.answer*probabilities/vols)
            # xpoints = np.linspace(0,1.5,100)
            # xpoints = np.array([np.array([x, *([0.]*(problem.D-1))]) for x in xpoints])
            # plt.plot(xpoints, problem.integrand(xpoints))
            # plt.show()

            # expand tree
            for k in range(len(containers)):
                containers[k].add(xs[samp == k], ys[samp == k])

            if i != N_iter-1:
                containers = self.construct_tree(containers)

            plt.hist([c.N for c in containers])
            plt.show()
            print(sum(np.array([c.volume for c in containers])*np.array([np.mean(c.y) for c in containers])))

            
        for iter in range(N_iter):
            print('I' + str(iter+1) + ' = ' + str(I_hats[iter]) + '    Sigma' + str(iter+1) +
                   ' = ' + str(np.sqrt(Sig2_Hats[iter])))
        
        # I = sum(I_hats / Sig2_Hats)/sum(1/Sig2_Hats)
        I = sum(I_hats)/N_iter
        S2 = 1/sum(1/Sig2_Hats)
        print('final results:  I = ' + str(I) + '   SigmaSquared = ' + str(np.sqrt(S2)))
        






        
        
        

