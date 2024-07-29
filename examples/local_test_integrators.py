import argparse, os

from treeQuadrature.compare_integrators import test_integrators
from treeQuadrature.exampleProblems import SimpleGaussian, Camel, QuadCamel
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator, VegasIntegrator, BayesMcIntegrator, SmcIntegrator

# Set up argument parser
parser = argparse.ArgumentParser(description="Run integrator tests with various configurations.")
parser.add_argument('--dimensions', type=int, nargs='+', default=[2], help='List of problem dimensions (default: [2])')
parser.add_argument('--threshold', type=float, default=0.5, help='Performance threshold for GpTQ (default: 0.5)')
parser.add_argument('--max_redraw', type=int, default=4, help='Max redraw attempts for RbfIntegral (default: 4)')
parser.add_argument('--n_splits', type=int, default=4, help='number of splits for K-fold CV in fitting GP (default: 4)')
parser.add_argument('--gp_range', type=float, default=50.0, help='Range of GP tuning for GpTreeIntegrator (default: 50.0)')
parser.add_argument('--grid_size', type=float, default=0.05, help='Range of GP tuning for GpTreeIntegrator (default: 0.05)')
parser.add_argument('--base_N', type=int, default=8000, help='Base sample size for integrators (default: 8000)')
parser.add_argument('--lsi_base_N', type=int, default=300, help='Base sample size for LimitedSampleIntegrator (default: 300)')
parser.add_argument('--lsi_active_N', type=int, default=10, help='active sample size for LimitedSampleIntegrator (default: 10)')
parser.add_argument('--bmc_N', type=int, default=400, help='Base sample size for BMC (default: 400)')
parser.add_argument('--P', type=int, default=50, help='Size of the largest container (default: 50)')
parser.add_argument('--vegas_iter', type=int, default=20, help='Number of iterations for VegasIntegrator (default: 20)')
parser.add_argument('--N_bmc', type=int, default=400, help='Sample size for BayesMcIntegrator (default: 400)')
parser.add_argument('--max_time', type=float, default=150.0, help='Maximum allowed time for each integrator (default: 150.0)')
parser.add_argument('--n_repeat', type=int, default=5, help='Number of repetitions for each test (default: 5)')

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

rbfIntegral = RbfIntegral(max_redraw=args.max_redraw, threshold=args.threshold, n_splits=args.n_splits)
rbfIntegral_2 = RbfIntegral(range=args.gp_range, max_redraw=args.max_redraw, threshold=args.threshold, n_splits=args.n_splits)
ranIntegral = RandomIntegral()

### Splits
split = MinSseSplit()

### Integrators
integ1 = SimpleIntegrator(base_N=args.base_N, P=args.P, 
                          split=split, integral=rbfIntegral)
integ1.name = 'TQ with RBF'

integ2 = GpTreeIntegrator(base_N=args.base_N, P=args.P, split=split, 
                          integral=rbfIntegral_2, grid_size=args.grid_size)
integ2.name = 'GpTQ with RBF'

integ3 = LimitedSampleIntegrator(int(args.base_N/4), base_N=args.lsi_base_N, 
                                 active_N=args.lsi_active_N, 
                                 split=split, integral=ranIntegral, 
                                 weighting_function=lambda container: container.volume)
integ3.name = 'LimitedSampleIntegrator'

integ4 = VegasIntegrator(int(args.base_N / args.vegas_iter) * 2, args.vegas_iter)
integ4.name = 'Vegas'

integ5 = BayesMcIntegrator(N = args.bmc_N)
integ5.name = 'BMC'

integ6 = SmcIntegrator(N=args.base_N)
integ6.name = 'SMC'

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 
                           f"../results/local_results_{'_'.join(map(str, Ds))}D_{args.n_repeat}repeat.csv")

if __name__ == '__main__':
    test_integrators([integ1, integ2, integ3, integ4, integ5, integ6],
                    problems=problems, 
                    output_file=output_path,
                    max_time=args.max_time, n_repeat=args.n_repeat) 