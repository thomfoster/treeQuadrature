import wandb
import sys
import argparse

import numpy as np
import treeQuadrature as tq

from tqdm import tqdm
from datetime import datetime


########################
# Input processing
########################

parser = argparse.ArgumentParser(
    description='Run a simple monte carlo integration method over dimensions 1,...,max_d.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Method specific args
parser.add_argument('--N', type=int, default=100_000, help="Number of samples to draw, in advance, from the distribution.")

# Method agnostic args
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

if args.problem == 'SimpleGaussian':
    problem = tq.exampleProblems.SimpleGaussian
elif args.problem == 'Camel':
    problem = tq.exampleProblems.Camel
elif args.problem == 'QuadCamel':
    problem = tq.exampleProblems.QuadCamel
else:
    raise Exception(f'Specified problem {args.problem} is not recognised - ensure CaptialisedCamelCase')


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

# Set up experiment on wandb
wandb.init(project=args.wandb_project)

# Log args
wandb.config.update(vars(args))

for D in tqdm(Ds):
    problem_instance = problem(D)
    integ = tq.integrators.SmcIntegrator(args.N)
    experiment(problem_instance, integ)