import wandb
import numpy as np
import sys

import numpy as np
from problems import SimpleGaussian
from treeQuadrature import Container
from treeQuadrature.splits import kdSplit, minSseSplit
from treeQuadrature.containerIntegration import midpointIntegral, randomIntegral
from treeQuadrature.visualisation import plotContainer
from treeQuadrature.utils import scale
from queue import SimpleQueue
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime

Ds = list(range(1,11))  # dimensions to test
N = 20_000
TOTAL_SPLITS = 6000
P = 3
accentuation_factor = 50

class ReservoirQueue:
    def __init__(self, accentuation_factor=1):
        '''The higher the accentuation factor the more like a priority queue it is'''
        self.items = []
        self.weights = []
        self.n = 0
        self.accentuation_factor = accentuation_factor

    def put(self, item, weight):
        self.items.append(item)
        self.weights.append(weight)
        self.n += 1

    def get(self):
        if self.n == 0:
            return None
        else:
            min_val = min(self.weights)
            range_val = 1 + max(self.weights) - min_val
            f = lambda w : ((w - min_val + 1)/range_val)**self.accentuation_factor
            probabilities = [f(weight) for weight in self.weights]
            sum_weights = sum(probabilities)
            probabilities = [p / sum_weights for p in probabilities]
            choice_of_index = np.random.choice(list(range(len(self.items))), p=probabilities)
            choice = self.items.pop(choice_of_index)
            chosen_weight = self.weights.pop(choice_of_index)
            self.n -= 1
            return choice
    
    def empty(self):
        return self.n == 0


# Define the Integrator we gonna be testing today
# Define the Integrator we gonna be testing today
class SimpleIntegrator:
    '''
    A simple integrator has the following pattern:
        - Draw <N> samples
        - Keep performing <split> method on containers...
        - ...until each container has less than <P> samples
        - Then perform  <integral> on each container and sum.
    '''
    
    def __init__(self, N, P, split, integral):
        self.N = N
        self.P = P
        self.split = split
        self.integral = integral
        
    def __call__(self, problem):
        D = problem.D
        
        # Draw samples
        X = problem.d.rvs(self.N)
        y = problem.pdf(X)
        
        root = Container(X, y, mins=[problem.low]*D, maxs=[problem.high]*D)
        
        # Construct tree
        n_splits = 0
        finished_containers = []
        q = ReservoirQueue(accentuation_factor=accentuation_factor)
        q.put(root,1)

        while (not q.empty()) and (n_splits < TOTAL_SPLITS):

            c = q.get()

            if c.N <= self.P:
                finished_containers.append(c)
                
            else:
                children = self.split(c)
                for child in children:
                    weight = np.max(child.y) - np.min(child.y)
                    q.put(child, weight)
                n_splits += 1
        
        # Empty queue
        while not q.empty():
            finished_containers.append(q.get())

        # Integrate containers
        contributions = [self.integral(cont, problem.pdf) for cont in finished_containers]
        G = np.sum(contributions)

        print(f'Finished in {n_splits} splits')
        
        return G
    

# Define the test for each dimension
def experiment(problem, integ):
    start_time = datetime.now()
    I_hat = integ(problem)
    end_time = datetime.now()
    
    d = {}
    d['D'] = problem.D
    d['N'] = integ.N
    d['pcntError'] = 100 * (I_hat - problem.answer) / problem.answer
    d['time'] =  (end_time - start_time).total_seconds()
    
    wandb.log(d)
    
def main():
    assert len(sys.argv) == 2, 'Usage: thisScript.py groupKey'
    
    # Set up experiment
    wandb.init(project="BoilerPlate")

    # Params for config logging
    wandb.config.Ds = Ds
    wandb.config.key = str(sys.argv[1])
    wandb.config.N = N
    wandb.config.TOTAL_SPLITS = TOTAL_SPLITS
    wandb.config.P = P
    wandb.config.split = str(minSseSplit)
    wandb.config.integral = str(randomIntegral)
    wandb.config.accentuation_factor = accentuation_factor

    # Run the experiment
    for D in Ds:
        problem = SimpleGaussian(D)
        split_partial = partial(minSseSplit, f=problem.pdf)
        integ_partial = partial(randomIntegral, n=50)
        integ = SimpleIntegrator(N, P, split_partial, integ_partial)
        experiment(problem, integ)
        
if __name__ == '__main__':
    main()