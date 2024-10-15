from treeQuadrature.integrators import DistributedTreeIntegrator
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.container_integrators import RandomIntegral
from treeQuadrature.trees import SimpleTree
from treeQuadrature.splits import MinSseSplit, relative_sse_score, sse_score
from treeQuadrature.example_problems import Camel, Ripple
from treeQuadrature.compare_integrators import test_integrators

import os
import argparse


parser = argparse.ArgumentParser(description="Compare different settings of MinSSE split.")
parser.add_argument('--max_time', type=float, default=180.0,
                    help='Maximum allowed time for each integrator (default: 180.0)')
parser.add_argument('--n_repeat', type=int, default=5,
                    help='Number of repetitions for each test (default: 5)')
args = parser.parse_args()
Ds = range(1, 16, 2)

mcmcSampler = McmcSampler()
rmeanIntegral = RandomIntegral()

# =======
# splits
# =======
split_default_sse = MinSseSplit(scoring_function=sse_score)
split_default_sse_random = MinSseSplit(scoring_function=sse_score,
                                       random_selection=True)
split_rel_sse = MinSseSplit(scoring_function=relative_sse_score)
split_rel_sse_random = MinSseSplit(scoring_function=relative_sse_score,
                                   random_selection=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    for D in Ds:
        output_path = os.path.join(
            script_dir,
            f"../test_results/sse_split/results_{D}D_{args.n_repeat}repeat.csv")

        max_n_samples = int(60_000 * (D/2))
        N = int(30_000 * (D/3))
        min_container_samples = 20
        max_container_samples = 500

        problems = [Camel(D), Ripple(D)]

        # default SSE score, no randomness in choosing the split dimensions
        integ_default_sse = DistributedTreeIntegrator(
            N, max_n_samples=max_n_samples,
            integral=rmeanIntegral, sampler=mcmcSampler,
            tree=SimpleTree(split=split_default_sse),
            max_container_samples=max_container_samples,
            min_container_samples=min_container_samples)
        integ_default_sse.name = 'TQ using SSE score, no randomness'

        # default SSE score, with randomness in choosing the split dimensions
        integ_default_sse_random = DistributedTreeIntegrator(
            N, max_n_samples=max_n_samples,
            integral=rmeanIntegral, sampler=mcmcSampler,
            tree=SimpleTree(split=split_default_sse_random),
            max_container_samples=max_container_samples,
            min_container_samples=min_container_samples)
        integ_default_sse_random.name = 'TQ using SSE score, with randomness'

        # relative SSE score, no randomness in choosing the split dimensions
        integ_rel_sse = DistributedTreeIntegrator(
            N, max_n_samples=max_n_samples,
            integral=rmeanIntegral, sampler=mcmcSampler,
            tree=SimpleTree(split=split_rel_sse),
            max_container_samples=max_container_samples,
            min_container_samples=min_container_samples)
        integ_rel_sse.name = 'TQ using relative SSE score, no randomness'

        # relative SSE score, no randomness in choosing the split dimensions
        integ_rel_sse_random = DistributedTreeIntegrator(
            N, max_n_samples=max_n_samples,
            integral=rmeanIntegral, sampler=mcmcSampler,
            tree=SimpleTree(split=split_rel_sse_random),
            max_container_samples=max_container_samples,
            min_container_samples=min_container_samples)
        integ_rel_sse_random.name = 'TQ using relative SSE score, with randomness'

        test_integrators([
            integ_default_sse, integ_default_sse_random, integ_rel_sse, integ_rel_sse_random],
            problems=problems,
            output_file=output_path,
            max_time=args.max_time, n_repeat=args.n_repeat)