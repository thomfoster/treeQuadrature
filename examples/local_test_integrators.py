import argparse, os, json
import numpy as np 

from treeQuadrature.compare_integrators import test_integrators
from treeQuadrature.exampleProblems import SimpleGaussian, Camel, QuadCamel, ExponentialProductProblem, QuadraticProblem, RippleProblem, OscillatoryProblem, ProductPeakProblem, CornerPeakProblem, DiscontinuousProblem, C0Problem
from treeQuadrature.containerIntegration import RandomIntegral, IterativeRbfIntegral, AdaptiveRbfIntegral
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.integrators import DistributedSampleIntegrator, LimitedSampleIntegrator, VegasIntegrator, BayesMcIntegrator, SmcIntegrator
from treeQuadrature.samplers import McmcSampler, LHSImportanceSampler, UniformSampler, ImportanceSampler

# Set up argument parser
parser = argparse.ArgumentParser(description="Run integrator tests with various configurations.")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help='List of problem dimensions (default: [2])')
parser.add_argument('--n_samples', type=int, default=20, help='number of samples drawn from each container (default: 20)')
parser.add_argument('--base_N', type=int, default=10_000, help='Base sample size for integrators when D = 3 (default: 10_000)')
parser.add_argument('--max_samples', type=int, default=15_000, help='Maximum sample size when D = 3 (default: 15_000)')
parser.add_argument('--gp_max_container_samples', type=int, default=150, help='maximum sample size for a container when fitting GP (default: 150)')
parser.add_argument('--lsi_base_N', type=int, default=1_000, help='Base sample size for LimitedSampleIntegrator (default: 1_000)')
parser.add_argument('--lsi_active_N', type=int, default=10, help='active sample size for LimitedSampleIntegrator (default: 10)')
parser.add_argument('--bmc_N', type=int, default=1200, help='Base sample size for BMC (default: 1200)')
parser.add_argument('--P', type=int, default=40, help='Size of the largest container (default: 40)')
parser.add_argument('--vegas_iter', type=int, default=40, help='Number of iterations for VegasIntegrator (default: 40)')
parser.add_argument('--max_time', type=float, default=180.0, help='Maximum allowed time for each integrator (default: 180.0)')
parser.add_argument('--n_repeat', type=int, default=5, help='Number of repetitions for each test (default: 5)')
parser.add_argument('--sampler', type=str, default='mcmc', help="The sampler used, must be one of 'mcmc', 'lhs', 'is' (default: mcmc)")
parser.add_argument('--split', type=str, default='minsse', help="The splitting method used, must be one of 'minsse', 'kd' (default: 'minsse')")

# receive arguments from parser
args = parser.parse_args()

### define the problems
Ds = args.dimensions
               
### container Integrals 
ranIntegral = RandomIntegral(n_samples=args.n_samples)
aRbf = AdaptiveRbfIntegral(n_samples= args.n_samples, max_redraw=0)

### Splits
if args.split == 'minsse':
    split = MinSseSplit()
elif args.split == 'kd':
    split = KdSplit()

### Sampler
if args.sampler == 'mcmc':
    sampler = McmcSampler()
elif args.sampler == 'lhs':
    sampler = LHSImportanceSampler()
elif args.sampler == 'unif':
    sampler = UniformSampler()
elif args.sampler == 'is':
    sampler = ImportanceSampler()
else:
    raise ValueError("Sampler must be one of 'mcmc', 'sobol', 'is', 'unif'")

## change base_N and max_n_samples with dimension
def get_base_N(D):
    return args.base_N * (D / 3) 

def get_max_n_samples(D):
    return args.max_samples * D * (D / 3) 

script_dir = os.path.dirname(os.path.abspath(__file__))
location_prefix = 'fourth_run/'
location_postfix = ''

if __name__ == '__main__':
    args_dict = vars(args)
    del args_dict['dimensions']
    # will be redistributed by integrator anyway
    del args_dict['n_samples']
    
    for D in Ds:
        ### remember to change file path for new run of experiments
        output_path = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}results_{D}D_{args.n_repeat}repeat{location_postfix}.csv")
        config_path = os.path.join(script_dir, 
                                f"../test_results/{location_prefix}configs_{D}D_{args.n_repeat}repeat{location_postfix}.json")

        base_N = get_base_N(D)
        max_samples = get_max_n_samples(D)

        args_dict['max_samples'] = max_samples
        args_dict['base_N'] = base_N

        with open(config_path, 'w') as file:
            json.dump(args_dict, file, indent=4)

        problems = [
            SimpleGaussian(D),
            Camel(D),
            QuadCamel(D),
            ExponentialProductProblem(D),
            QuadraticProblem(D),
            RippleProblem(D),
            OscillatoryProblem(D, a=np.array(10 / np.linspace(1, D, D))),
            ProductPeakProblem(D, a=10),
            C0Problem(D, a=1.1),
            CornerPeakProblem(D, a=10)
        ]

        integ_simple = DistributedSampleIntegrator(base_N, args.P, max_samples, split, ranIntegral, 
                                                sampler=sampler)
        integ_simple.name = 'TQ with mean'
        
        lsi_N = int(max_samples / (args.n_samples + 1)) * 1.3
        integ_active = LimitedSampleIntegrator(N=lsi_N, base_N=base_N, 
                                        active_N=args.lsi_active_N, 
                                        split=split, integral=ranIntegral, 
                                        weighting_function=lambda container: container.volume, 
                                        sampler=sampler)
        integ_activeTQ = DistributedSampleIntegrator(base_N, args.P, max_samples, 
                                                    split, ranIntegral, sampler=sampler,
                                                    construct_tree_method=integ_active.construct_tree)
        integ_activeTQ.name = 'ActiveTQ'
        
        integ_rbf = DistributedSampleIntegrator(base_N, args.P, max_samples, split, aRbf, sampler=sampler, 
                                                min_n_samples=10)
        integ_rbf.name = 'TQ with Rbf'
        
        n_iter = args.vegas_iter
        n_vegas = int(max_samples / n_iter)
        integ_vegas= VegasIntegrator(n_vegas, n_iter)
        integ_vegas.name = 'Vegas'
        
        integ_bmc = BayesMcIntegrator(N=args.bmc_N, sampler=sampler)
        integ_bmc.name = 'BMC'
        
        integ_smc = SmcIntegrator(N=max_samples, sampler=UniformSampler())
        integ_smc.name = 'SMC'

        # Now run the tests for the current dimension D
        test_integrators([integ_simple, integ_activeTQ, integ_rbf, integ_smc, integ_vegas],
                        problems=problems, 
                        output_file=output_path,
                        max_time=args.max_time, n_repeat=args.n_repeat, 
                        integrator_specific_kwargs={'ActiveTQ': {'integrand' : None}})