import wandb
import sys
import argparse

import numpy as np
import treeQuadrature as tq

from functools import partial
from datetime import datetime
from tqdm import tqdm


########################
# Input processing
########################

parser = argparse.ArgumentParser(
    description='Run the simpleIntegrator treeQuadrature method over dimensions 1,...,max_d.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Method specific args
parser.add_argument('--N', type=int, default=100_000, help="Max sample limit")
parser.add_argument('--base_N', type=int, default=50_000, help="Number of samples to draw, in advance, from the distribution.")
parser.add_argument('--active_N', type=int, default=0, help="Number of uniformly distributed samples to draw over each container at each splitting step")
parser.add_argument('--split', type=str, default='kdSplit', help="Method used to split containers")
parser.add_argument('--integral', type=str, default='midpointIntegral', help="Method used to integrate containers")
parser.add_argument('--weighting_function', type=str, default='yvar', help="name of function used to assign the probability weight for a container")
parser.add_argument('--queue', type=str,  default="PriorityQueue", help="which type of queue to use out of [PriorityQueue, ReservoirQueue]")
parser.add_argument('--accentuation_factor', type=int, default=100, help="weights in the reservoir queue are raised to this power. The higher the value the more the reservoir becomes like a priority queue")
parser.add_argument('--num_extra_samples', type=int, default=100, help="If randomIntegral is selected, this is the number of extra samples that this method draws uniformly over the container to integrate it. Else unused.")

# Method agnostic args
parser.add_argument('--problem', type=str, default='SimpleGaussian', help='The problem to test on')
parser.add_argument('--max_d', type=int, default=10, help="Maximum dimension to test the integrator in")
parser.add_argument('--key', type=str, default='', help='Key to submit to weights and biases that can be used to group this run with other runs')
parser.add_argument('--wandb_project', type=str, default='BoilerPlate', help='Weights and Bias project to log results to')

args = parser.parse_args()

Ds = list(range(1, args.max_d + 1))

if args.wandb_project == "by_problem_name":
    args.wandb_project = args.problem

if args.problem == 'SimpleGaussian':
    problem = tq.example_problems.SimpleGaussian
elif args.problem == 'Camel':
    problem = tq.example_problems.Camel
elif args.problem == 'QuadCamel':
    problem = tq.example_problems.QuadCamel
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
elif args.integral == 'smcIntegral':
    integral = partial(tq.containerIntegration.smcIntegral, n=args.num_extra_samples)
else:
    raise Exception(f'Integral method {args.integral} not recognised')

if args.weighting_function == 'yvar':
    def safe_var(y):
        if y.shape[0] < 2:
            return 0
        else:
            return np.max(np.var(y, axis=0))
    weighting_function = lambda c: safe_var(c.y)
elif args.weighting_function == 'volume':
    weighting_function = lambda c: c.volume
elif args.weighting_function == 'range':
    weighting_function = lambda c: np.max(c.y) - np.min(c.y) + 1
else:
    raise Exception(f'Weighting method {args.weighting_function} not recognised')


if args.queue == 'PriorityQueue':
    queue = tq.queues.PriorityQueue
elif args.queue == 'ReservoirQueue':
    queue = partial(tq.queues.ReservoirQueue, accentuation_factor=args.accentuation_factor)
else:
    raise Exception(f"Queue type {args.queue} not recognised")


    
##########################################
# Define the test for each dimension
# - largerly the same for any integrator
##########################################

def experiment(problem, integ):
    start_time = datetime.now()
    I_hat, N = integ(problem, return_N=True)
    end_time = datetime.now()
    
    d = {}
    d['D'] = problem.D
    d['N'] = N
    d['pcntError'] = 100 * (I_hat - problem.answer) / problem.answer
    d['time'] =  (end_time - start_time).total_seconds()
    
    wandb.log(d)
    

#######################################################################
# Start of script
#######################################################################

# Set up experiment
wandb.init(project=args.wandb_project)

# Log args
wandb.config.update(vars(args))

for D in tqdm(Ds):
    problem_instance = problem(D)
    tree = tq.trees.LimitedSampleTree(N=args.N, active_N=args.active_N, 
                                      split=split, weighting_function=weighting_function, 
                                      queue=queue)
    integ = tq.integrators.TreeIntegrator(args.base_N, integral=integral, tree = tree)
    experiment(problem_instance, integ)