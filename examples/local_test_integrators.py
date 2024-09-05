import argparse, os, json
import numpy as np 

from treeQuadrature.compare_integrators import test_integrators
from treeQuadrature.example_problems import (
    SimpleGaussian, Camel, QuadCamel, 
    ExponentialProduct, Quadratic, Ripple,
    Oscillatory, ProductPeak, CornerPeak,
    Discontinuous, C0
)
from treeQuadrature.container_integrators import RandomIntegral, AdaptiveRbfIntegral
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.integrators import DistributedTreeIntegrator, VegasIntegrator, VegasTreeIntegrator
from treeQuadrature.samplers import McmcSampler, LHSImportanceSampler, UniformSampler, ImportanceSampler
from treeQuadrature.trees import LimitedSampleTree, SimpleTree

# Set up argument parser
parser = argparse.ArgumentParser(description="Run integrator tests with various configurations.")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help='List of problem dimensions (default: [2])')
parser.add_argument('--n_samples', type=int, default=20, help='number of samples drawn from each container (default: 20)')
parser.add_argument('--base_N', type=int, default=30_000, help='Base sample size for integrators when D = 3 (default: 30_000)')
parser.add_argument('--max_samples', type=int, default=60_000, help='Maximum sample size when D = 2 (default: 60_000)')
parser.add_argument('--min_container_samples', type=int, default=32, help='minimum sample size for a container when fitting GP (default: 32)')
parser.add_argument('--max_container_samples', type=int, default=600, help='maximum sample size for a container when fitting GP (default: 800)')
parser.add_argument('--lsi_base_N', type=int, default=6_000, help='Base sample size for LimitedSampleTree when D = 3 (default: 1_000)')
parser.add_argument('--lsi_active_N', type=int, default=0, help='active sample size for LimitedSampleTree (default: 0)')
parser.add_argument('--P', type=int, default=50, help='Size of the largest container (default: 50)')
parser.add_argument('--vegas_iter', type=int, default=10, help='Number of iterations for VegasIntegrator (default: 10)')
parser.add_argument('--max_time', type=float, default=180.0, help='Maximum allowed time for each integrator (default: 180.0)')
parser.add_argument('--n_repeat', type=int, default=10, help='Number of repetitions for each test (default: 10)')
parser.add_argument('--sampler', type=str, default='mcmc', help="The sampler used, must be one of 'mcmc', 'lhs', 'is' (default: mcmc)")
parser.add_argument('--split', type=str, default='minsse', help="The splitting method used, must be one of 'minsse', 'kd' (default: 'minsse')")
parser.add_argument('--retest', type=str, nargs='+', default=[], help='List of integrators to be retested (default: [])')
parser.add_argument('--file_location', type=str, default='', help='file location prefix (default: saved to test_results/)')

# receive arguments from parser
args = parser.parse_args()

# change base_N and max_n_samples with dimension
def get_base_N(D) -> int:
    return max(int(args.base_N * (D / 3)), 6000)

def get_max_n_samples(D) -> int:
    return max(int(args.max_samples * (D / 2)), 12_000)

def get_lsi_base_N(D) -> int:
    return int(args.lsi_base_N * (D / 3))

def get_max_iter(D) -> int:
    return int(2000 + D * 500)

def volume_weighting_function(container):
    return container.volume


if __name__ == '__main__':
    ### define the problems
    Ds = args.dimensions
                
    ### container Integrals 
    ranIntegral = RandomIntegral(n_samples=args.n_samples)
    aRbf = AdaptiveRbfIntegral(n_samples= args.n_samples)

    # =======
    # Splits
    # =======
    if args.split == 'minsse':
        split = MinSseSplit()
    elif args.split == 'kd':
        split = KdSplit()
    else:
        raise ValueError("Only 'minsse' and 'kd' supported")

    # =======
    # Sampler
    # =======
    if args.sampler == 'mcmc':
        sampler = McmcSampler()
    elif args.sampler == 'lhs':
        sampler = LHSImportanceSampler()
    elif args.sampler == 'unif':
        sampler = UniformSampler()
    elif args.sampler == 'is':
        sampler = ImportanceSampler()
    else:
        raise ValueError(
            "Sampler must be one of 'mcmc', 'sobol', 'is', 'unif'")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    location_prefix = args.file_location

    args_dict = vars(args).copy()
    del args_dict['dimensions']
    # will be redistributed by integrator anyway
    del args_dict['n_samples']
    
    for D in Ds:
        ### remember to change file path for new run of experiments
        output_path = os.path.join(
            script_dir,
            f"../test_results/{location_prefix}results_{D}D_{args.n_repeat}repeat.csv")
        config_path = os.path.join(
            script_dir,
            f"../test_results/{location_prefix}configs_{D}D_{args.n_repeat}repeat.json")

        base_N = get_base_N(D)
        max_samples = get_max_n_samples(D)
        lsi_base_N = get_lsi_base_N(D)
        lsi_N = int(max_samples / (args.n_samples + 1))
        max_iter = get_max_iter(D)

        args_dict['max_samples'] = max_samples
        args_dict['base_N'] = base_N
        args_dict['lsi_base_N'] = lsi_base_N
        args_dict['lsi_N'] = lsi_N
        args_dict['max_iter'] = lsi_N

        with open(config_path, 'w') as file:
            json.dump(args_dict, file, indent=4)

        if D > 6:
            problems = [
                SimpleGaussian(D),
                Camel(D),
                ExponentialProduct(D),
                Quadratic(D),
                Ripple(D),
                Oscillatory(D, a=np.array(10 / np.linspace(1, D, D))),
                C0(D, a=1.1),
                ProductPeak(D, a=10),
                CornerPeak(D, a=10), 
                Discontinuous(D, a=10)
            ]
        else:
            problems = [
                SimpleGaussian(D),
                Camel(D),
                QuadCamel(D),
                ExponentialProduct(D),
                Quadratic(D),
                Ripple(D),
                Oscillatory(D, a=np.array(10 / np.linspace(1, D, D))),
                ProductPeak(D, a=10),
                C0(D, a=1.1),
                CornerPeak(D, a=10), 
                Discontinuous(D, a=10)
            ]

        # =======
        # Trees
        # =======
        tree_active = LimitedSampleTree(
        N=lsi_N, active_N=args.lsi_active_N, split=split,
        weighting_function=volume_weighting_function,
        max_iter=max_iter * 2) # allow twice more splits
        tree_simple = SimpleTree(
            split, args.P, max_iter=max_iter)

        # ===========
        # Integrators
        # ===========
        integ_simple = DistributedTreeIntegrator(
            base_N, max_samples, ranIntegral,
            sampler=sampler, tree=tree_simple,
            min_container_samples=args.min_container_samples,
            max_container_samples=args.max_container_samples)
        integ_simple.name = 'TQ with mean'
        
        # half min samples to compensate
        integ_activeTQ = DistributedTreeIntegrator(
            lsi_base_N, max_samples, ranIntegral, sampler=sampler,
            tree=tree_active, min_container_samples=args.min_container_samples // 2,
            max_container_samples=args.max_container_samples)
        integ_activeTQ.name = 'ActiveTQ'
        
        integ_rbf = DistributedTreeIntegrator(
            base_N, max_samples, aRbf,
            sampler=sampler, tree=tree_simple,
            min_container_samples=args.min_container_samples,
            max_container_samples=args.max_container_samples)
        integ_rbf.name = 'TQ with Rbf'
        
        n_iter = args.vegas_iter
        # 5 accounts for adaptive iterations
        n_vegas = int(max_samples / (n_iter + 5))
        integ_vegas= VegasIntegrator(n_vegas, n_iter)
        integ_vegas.name = 'Vegas'

        integ_vegas_tree_rbf = VegasTreeIntegrator(
            base_N, tree=tree_simple,
            integral=aRbf,
            max_N=max_samples,
            min_container_samples=args.min_container_samples,
            max_container_samples=args.max_container_samples,
            vegas_iter=n_iter+5)
        integ_vegas_tree_rbf.name = 'Vegas + TQ + RBF'

        # Now run the tests for the current dimension D
        test_integrators([
            integ_simple, integ_activeTQ, integ_rbf, integ_vegas, integ_vegas_tree_rbf],
            problems=problems,
            output_file=output_path,
            max_time=args.max_time, n_repeat=args.n_repeat,
            retest_integrators=args.retest)