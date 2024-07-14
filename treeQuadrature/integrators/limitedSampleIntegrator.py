import matplotlib.pyplot as plt
import numpy as np

from ..queues import ReservoirQueue
from ..container import Container

# Default finished condition will never prevent container being split


def default_stopping_condition(container): return False


default_queue = ReservoirQueue(accentuation_factor=100)


def save_weights_image(q):
    weights = q.weights
    ps = q.get_probabilities(weights)
    plt.figure()
    plt.yscale("log")
    plt.hist(ps)
    plt.savefig(
        "/home/t/Documents/4yp/evidence-with-kdtrees/" +
        "src/treeQuadrature/results/images/ps_"
        + str(q.n) + ".png")
    plt.close()


class LimitedSampleIntegrator:
    """
    Integrator that builds on from queueIntegrator with more friendly
    controls - just keeps sampling until all samples used up.
    User does not need to specify the stopping condition

    Parameters
    ----------
    N : int
        Total number of samples to use.
    base_N : int
        Number of base samples.
    active_N : int
        Number of active samples per iteration.
    split : function
        Function to split a container into sub-containers.
    integral : function
        Function to compute the integral over a container.
    weighting_function : function
        Function to compute the weight of a container.
    queue : class
        Queue class to manage the containers, default is PriorityQueue.
    """

    def __init__(
            self,
            N,
            base_N,
            active_N,
            split,
            integral,
            weighting_function,
            queue=default_queue):
        
        self.N = N
        self.base_N = base_N
        self.active_N = active_N
        self.split = split
        self.integral = integral
        self.weighting_function = weighting_function
        self.queue = queue

    def __call__(self, problem, return_N=False, return_all=False):
        """
        Perform the integration process.

        Arguments
        ----------
        problem : Problem
            The integration problem to be solved
        return_N : bool, optional
            If True, return the number of samples used.
        return_all : bool, optional
            If True, return containers and their contributions to the integral

        Returns
        -------
        result : tuple or float
            The computed integral and optionally the number of samples, finished containers, contributions, and remaining samples.
        """
        D = problem.D

        # Draw samples
        X = problem.d.rvs(self.base_N)
        y = problem.pdf(X)

        root = Container(X, y, mins=[problem.low] * D, maxs=[problem.high] * D)

        # Refine with further active samples
        q = self.queue()
        q.put(root, 1)
        finished_containers = []
        num_samples_left = self.N - self.base_N

        while not q.empty():

            # save_weights_image(q)

            c = q.get()

            if num_samples_left >= self.active_N:
                X = c.rvs(self.active_N)
                y = problem.pdf(X)
                c.add(X, y)
                num_samples_left -= self.active_N

            elif c.N < 2:
                finished_containers.append(c)
                continue

            children = self.split(c)
            for child in children:
                weight = self.weighting_function(child)
                q.put(child, weight)

        # Integrate containers
        contributions = [self.integral(cont, problem.pdf)
                         for cont in finished_containers]
        G = np.sum(contributions)
        N = sum([cont.N for cont in finished_containers])

        ret = (G, N) if return_N else G
        ret = (G, N, finished_containers, contributions,
               num_samples_left) if return_all else ret
        return ret
