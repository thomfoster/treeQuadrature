import argparse, os

from controlled_test import test_integrators
from treeQuadrature.exampleProblems import SimpleGaussian, Camel, QuadCamel
from treeQuadrature.containerIntegration import SmcIntegral, RbfIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator, VegasIntegrator, BayesMcIntegrator, SmcIntegrator

# Set up argument parser
parser = argparse.ArgumentParser(description="Run integrator tests with various configurations.")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help='List of problem dimensions (default: [2])')
parser.add_argument('--threshold', type=float, default=0.5, help='Performance threshold for GpTQ (default: 0.5)')
parser.add_argument('--max_redraw', type=int, default=4, help='Max redraw attempts for RbfIntegral (default: 4)')
parser.add_argument('--n_splits', type=int, default=4, help='number of splits for K-fold CV in fitting GP (default: 4)')
parser.add_argument('--n_samples', type=int, default=15, help='number of samples drawn from each container (default: 15)')
parser.add_argument('--gp_range', type=float, default=100.0, help='Range of GP tuning (default: 100.0)')
parser.add_argument('--base_N', type=int, default=10_000, help='Base sample size for integrators (default: 10_000)')
parser.add_argument('--lsi_N', type=int, default=2_000, help='Maximum sample size for LimitedSampleIntegrator (default: 2_000)')
parser.add_argument('--lsi_base_N', type=int, default=1_000, help='Base sample size for LimitedSampleIntegrator (default: 1_000)')
parser.add_argument('--lsi_active_N', type=int, default=10, help='active sample size for LimitedSampleIntegrator (default: 10)')
parser.add_argument('--bmc_N', type=int, default=1500, help='Base sample size for BMC (default: 1500)')
parser.add_argument('--P', type=int, default=40, help='Size of the largest container (default: 40)')
parser.add_argument('--vegas_iter', type=int, default=20, help='Number of iterations for VegasIntegrator (default: 20)')
parser.add_argument('--max_time', type=float, default=120.0, help='Maximum allowed time for each integrator (default: 120.0)')
parser.add_argument('--n_repeat', type=int, default=5, help='Number of repetitions for each test (default: 5)')
parser.add_argument('--ratio', type=float, default=2, help='Ratio of base_N between Rbf and Vegas, SMC')

# receive arguments from parser
args = parser.parse_args()

### define the problems
Ds = args.dimensions

problems = []
for D in Ds:
    problems.append(SimpleGaussian(D))
    problems.append(Camel(D))
    problems.append(QuadCamel(D))
               
### container Integrals 

rbfIntegral = RbfIntegral(range=args.gp_range, max_redraw=args.max_redraw, threshold=args.threshold, n_splits=args.n_splits, 
                          n_samples=args.n_samples)
ranIntegral = SmcIntegral(n=args.n_samples)
rbfIntegral_2 = RbfIntegral(range=args.gp_range, max_redraw=args.max_redraw, threshold=args.threshold, n_splits=args.n_splits, 
                            fit_residuals=False)

### Splits
split = MinSseSplit()

### Integrators
integ1 = SimpleIntegrator(base_N=args.base_N, P=args.P, 
                          split=split, integral=rbfIntegral_2)
integ1.name = 'TQ with RBF'

integ2 = SimpleIntegrator(base_N=args.base_N, P=args.P, 
                          split=split, integral=rbfIntegral)
integ2.name = 'TQ with RBF (mean)'

integ3 = LimitedSampleIntegrator(N=args.lsi_N, base_N=args.lsi_base_N, 
                                 active_N=args.lsi_active_N, 
                                 split=split, integral=ranIntegral, 
                                 weighting_function=lambda container: container.volume)
integ3.name = 'LimitedSampleIntegrator'

integ4 = VegasIntegrator(int(args.base_N / args.vegas_iter * args.ratio), args.vegas_iter)
integ4.name = 'Vegas'

integ5 = BayesMcIntegrator(N = args.bmc_N)
integ5.name = 'BMC'

integ6 = SmcIntegrator(N=int(args.base_N * args.ratio))
integ6.name = 'SMC'

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 
                           f"../test_results/results_{'_'.join(map(str, Ds))}D_{args.n_repeat}repeat_{args.base_N}base_N.csv")

if __name__ == '__main__':
    test_integrators([integ1, integ2, integ3, integ4, integ5, integ6],
                    problems=problems, 
                    output_file=output_path,
                    max_time=args.max_time, n_repeat=args.n_repeat) 