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
    description='Run the TreeIntegrator with a simple tree over dimensions 1,...,max_d.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--N', type=int, default=10_000, help="Number of samples to draw, in advance, from the distribution.")
parser.add_argument('--P', type=int, default=10, help="Stop splitting containers when they have less than P samples")
parser.add_argument('--split', type=str, default='kdSplit', help="Method used to split containers")
parser.add_argument('--integral', type=str, default='randomIntegral', help="Method used to integrate containers")
parser.add_argument('--num_extra_samples', type=int, default=10, help="If randomIntegral is selected, this is the number of extra samples that this method draws uniformly over the container to integrate it. Else unused.")

parser.add_argument('--problem', type=str, default='SimpleGaussian', help='The problem to test on')
parser.add_argument('--max_d', type=int, default=10, help="Maximum dimension to test the integrator in")
parser.add_argument('--key', type=str, default='', help='Key to submit to weights and biases that can be used to group this run with other runs')
parser.add_argument('--wandb_project', type=str, default='BoilerPlate', help='Weights and Bias project to log results to')
args = parser.parse_args()

print("Running with: ")
for k, v in vars(args).items():
    print(f'{k}: {v}')
print()

Ds = list(range(1, args.max_d + 1))

if args.wandb_project == "by_problem_name":
    args.wandb_project = args.problem

if args.problem == 'SimpleGaussian':
    problem = tq.exampleProblems.SimpleGaussian
elif args.problem == 'Camel':
    problem = tq.exampleProblems.Camel
elif args.problem == 'QuadCamel':
    problem = tq.exampleProblems.QuadCamel
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
   




#######################################
# Define the test for each dimension
# - largely the same for any integrator
#######################################

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
    integ = tq.integrators.TreeIntegrator(args.N, tree=tq.trees.SimpleTree(args.P, split), 
                                          integral=integral)
    experiment(problem_instance, integ)