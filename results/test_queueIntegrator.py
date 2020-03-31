import wandb
import sys
import argparse
import problems

import numpy as np
import treeQuadrature as tq

from queue import SimpleQueue
from functools import partial
from datetime import datetime


########################
# Input processing
########################

parser = argparse.ArgumentParser(
    description='Run the simpleIntegrator treeQuadrature method over dimensions 1,...,max_d.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--problem', type=str, default='SimpleGaussian', help='The problem to test on')
parser.add_argument('--N', type=int, default=100_000, help="Number of samples to draw, in advance, from the distribution.")
parser.add_argument('--P', type=int, default=10, help="Stop splitting containers when they have less than P samples")
parser.add_argument('--split', type=str, default='kdSplit', help="Method used to split containers")
parser.add_argument('--integral', type=str, default='midpointIntegral', help="Method used to integrate containers")
parser.add_argument('--total_splits', type=int, default=6000, help='limit the number of splits the integrator is allowed to perform')
parser.add_argument('--accentuation_factor', type=int, default=100, help="weights in the reservoir queue are raised to this power. The higher the value the more the reservoir becomes like a priority queue")
parser.add_argument('--max_d', type=int, default=10, help="Maximum dimension to test the integrator in")
parser.add_argument('--weight', type=str, default='yvar', help="name of function used to assign the probability weight for a container")
parser.add_argument('--key', type=str, default='', help='Key to submit to weights and biases that can be used to group this run with other runs')
parser.add_argument('--wandb_project', type=str, default='BoilerPlate', help='Weights and Bias project to log results to')
parser.add_argument('--num_extra_samples', type=int, default=100, help="If randomIntegral is selected, this is the number of extra samples that this method draws uniformly over the container to integrate it. Else unused.")
args = parser.parse_args()

Ds = list(range(1, args.max_d + 1))

if args.problem == 'SimpleGaussian':
    problem = problems.SimpleGaussian
elif args.problem == 'Camel':
    problem = problems.Camel
elif args.problem == 'QuadCamel':
    problem = problems.QuadCamel
else:
    raise Exception(f'Specified problem {args.problem} is not recognised - try CaptialisedCamelCase')

if args.split == 'kdSplit':
    split = tq.splits.kdSplit
elif args.split == 'minSseSplit':
    split = tq.splits.minSseSplit
else:
    raise Exception(f'Split method {args.split} not recognised')

if args.integral == 'midpointIntegral':
    integral = tq.containerIntegration.midpointIntegral
elif args.integral == 'medianIntegral':
    integral = tq.containerIntegration.medianIntegral
elif args.integral == 'randomIntegral':
    integral = partial(tq.containerIntegration.randomIntegral, n=args.num_extra_samples)
else:
    raise Exception(f'Integral method {args.integral} not recognised')

if args.weight == 'yvar':
    weight = lambda c: np.var(c.y)
elif args.weight == 'volume':
    weight = lambda c: c.volume
else:
    raise Exception(f'Weight method {args.weight} not recognised')


######################################################
# Define the Integrator we gonna be testing today
######################################################

class ReservoirIntegrator:
    
    def __init__(self, base_N, active_N, ):
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