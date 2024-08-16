import argparse, os, json
import numpy as np 

from treeQuadrature.compare_integrators import test_integrators
from treeQuadrature.exampleProblems import SimpleGaussian, Camel, QuadCamel, ExponentialProductProblem, QuadraticProblem, RippleProblem, OscillatoryProblem, ProductPeakProblem, CornerPeakProblem, DiscontinuousProblem, C0Problem
from treeQuadrature.containerIntegration import RandomIntegral, IterativeRbfIntegral
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.integrators import DistributedSampleIntegrator, LimitedSampleIntegrator, LimitedSamplesGpIntegrator, VegasIntegrator, BayesMcIntegrator, SmcIntegrator
from treeQuadrature.samplers import McmcSampler, SobolSampler, UniformSampler, ImportanceSampler

# Set up argument parser
parser = argparse.ArgumentParser(description="Run integrator tests with various configurations.")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help='List of problem dimensions (default: [2])')
parser.add_argument('--n_samples', type=int, default=20, help='number of samples drawn from each container (default: 20)')
parser.add_argument('--base_N', type=int, default=10_000, help='Base sample size for integrators (default: 10_000)')
parser.add_argument('--max_samples', type=int, default=30_000, help='Maximum sample size (default: 30_000)')
parser.add_argument('--lsi_base_N', type=int, default=1_000, help='Base sample size for LimitedSampleIntegrator (default: 1_000)')
parser.add_argument('--lsi_active_N', type=int, default=10, help='active sample size for LimitedSampleIntegrator (default: 10)')
parser.add_argument('--bmc_N', type=int, default=1200, help='Base sample size for BMC (default: 1200)')
parser.add_argument('--P', type=int, default=40, help='Size of the largest container (default: 40)')
parser.add_argument('--vegas_iter', type=int, default=40, help='Number of iterations for VegasIntegrator (default: 40)')
parser.add_argument('--max_time', type=float, default=180.0, help='Maximum allowed time for each integrator (default: 180.0)')
parser.add_argument('--n_repeat', type=int, default=5, help='Number of repetitions for each test (default: 5)')
parser.add_argument('--sampler', type=str, default='mcmc', help="The sampler used, must be one of 'mcmc', 'sobol', 'is', 'unif' (default: mcmc)")
parser.add_argument('--split', type=str, default='minsse', help="The splitting method used, must be one of 'minsse', 'kd' (default: 'minsse')")

# receive arguments from parser
args = parser.parse_args()

### define the problems
Ds = args.dimensions

problems = []
for D in Ds:
    problems.append(SimpleGaussian(D))
    problems.append(Camel(D))
    problems.append(QuadCamel(D))
    problems.append(ExponentialProductProblem(D))
    problems.append(QuadraticProblem(D))
    problems.append(RippleProblem(D))
    problems.append(OscillatoryProblem(D, a=np.array(10 / np.linspace(1, D, D))))
    problems.append(ProductPeakProblem(D, a=10))
    problems.append(C0Problem(D, a=1.1))
    problems.append(CornerPeakProblem(D, a=10))
               
### container Integrals 
ranIntegral = RandomIntegral(n_samples=args.n_samples)
iRbf = IterativeRbfIntegral(n_samples=args.n_samples)

### Splits
if args.split == 'minsse':
    split = MinSseSplit()
elif args.split == 'kd':
    split = KdSplit()

### Sampler
if args.sampler == 'mcmc':
    sampler = McmcSampler()
elif args.sampler == 'sobol':
    sampler = SobolSampler()
elif args.sampler == 'unif':
    sampler = UniformSampler()
elif args.sampler == 'is':
    sampler = ImportanceSampler()
else:
    raise ValueError("Sampler must be one of 'mcmc', 'sobol', 'is', 'unif'")


### Integrators
integ_simple = DistributedSampleIntegrator(args.base_N, args.P, args.max_samples, split, ranIntegral, 
                                           sampler=sampler)
integ_simple.name = 'TQ with mean'

lsi_N = int(args.max_samples / (args.n_samples + 1)) * 1.3
integ_active = LimitedSampleIntegrator(N=lsi_N, base_N=args.lsi_base_N, 
                                 active_N=args.lsi_active_N, 
                                 split=split, integral=ranIntegral, 
                                 weighting_function=lambda container: container.volume, 
                                 sampler=sampler)
integ_activeTQ = DistributedSampleIntegrator(args.base_N, args.P, args.max_samples, 
                                             split, ranIntegral, sampler=sampler,
                                             construct_tree_method=integ_active.construct_tree)
integ_activeTQ.name = 'ActiveTQ'

integ_limitedGp = LimitedSamplesGpIntegrator(base_N=args.base_N, P=args.P, 
                                             max_n_samples=args.max_samples,
                                            split=split, integral=iRbf, sampler=sampler, 
                                            max_container_samples=150)
integ_limitedGp.name = 'TQ with Rbf'

n_iter = args.vegas_iter
n_vegas = int(args.max_samples / n_iter)
integ_vegas= VegasIntegrator(n_vegas, n_iter)
integ_vegas.name = 'Vegas'

integ_bmc = BayesMcIntegrator(N = args.bmc_N, sampler=sampler)
integ_bmc.name = 'BMC'

integ_smc = SmcIntegrator(N=args.max_samples, sampler=UniformSampler())
integ_smc.name = 'SMC'

script_dir = os.path.dirname(os.path.abspath(__file__))
### remember to change file path for new run of experiments
output_path = os.path.join(script_dir, 
                           f"../test_results/fourth_run/results_{'_'.join(map(str, Ds))}D_{args.n_repeat}repeat_{args.base_N}base_N_{args.max_samples}max_samples.csv")
config_path = os.path.join(script_dir, 
                           f"../test_results/fourth_run/configs_{'_'.join(map(str, Ds))}D_{args.n_repeat}repeat_{args.base_N}base_N_{args.max_samples}max_samples.json")

if __name__ == '__main__':
    args_dict = vars(args)
    with open(config_path, 'w') as file:
        json.dump(args_dict, file, indent=4)
    test_integrators([integ_simple, integ_activeTQ, integ_limitedGp, integ_smc, integ_vegas],
                    problems=problems, 
                    output_file=output_path,
                    max_time=args.max_time, n_repeat=args.n_repeat, 
                    integrator_specific_kwargs={'ActiveTQ': {'integrand' : None}}) 