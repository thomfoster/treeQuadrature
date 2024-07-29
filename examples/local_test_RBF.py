from treeQuadrature.splits import MinSseSplit, UniformSplit, KdSplit
from treeQuadrature.integrators import SimpleIntegrator, VegasIntegrator, LimitedSampleIntegrator
from treeQuadrature.containerIntegration import RbfIntegral, RandomIntegral
from treeQuadrature.exampleProblems import Camel, SimpleGaussian, Gaussian, QuadCamel
from treeQuadrature.visualisation import plotContainers
import numpy as np

from treeQuadrature.compare_integrators import compare_integrators

# Problems
problem_simple_Gaussian = SimpleGaussian(D=3)
problem_large_Gaussian = Gaussian(D=2, lows=-5.0, highs=5.0)
problem_infinite_Gaussian = Gaussian(D=2, highs=1.0)
cov = np.array([[1, 0.3], [0.3, 1]])
problem_Gaussian_with_cov = Gaussian(D=2, lows=-1.0, highs=1.0, Sigma=cov)
problem_Camel = Camel(D=2)
problem_QuadCamel = QuadCamel(D=2)

# Non-active Integrators
N = 2_000
P = 40

rbfIntegral = RbfIntegral(n_samples=20, return_std=False)
minSseSplit = MinSseSplit()
integ_rbf = SimpleIntegrator(N, P, minSseSplit, rbfIntegral)
integ_rbf.name = 'RBF integrator with TQ'

randomIntegral = RandomIntegral()
integ = SimpleIntegrator(N, P, minSseSplit, randomIntegral)
integ.name = 'constant integrator with TQ'

integ_vegas = VegasIntegrator(100, 20)
integ_vegas.name = 'VEGAS'

# Active Integrators
volume_weighting = lambda container: container.volume
integ_queue = LimitedSampleIntegrator(
    N=2_000, base_N=500, active_N=10, split=minSseSplit, integral=randomIntegral, 
    weighting_function=volume_weighting
)
integ_queue.name = 'constant integrator with active TQ'

rbfIntegral_small = RbfIntegral(n_samples=10)
integ_queue_rbf = LimitedSampleIntegrator(
    N=200, base_N=100, active_N=10, split=minSseSplit, integral=rbfIntegral_small, 
    weighting_function=volume_weighting
)
integ_queue_rbf.name = 'RBF integrator with active TQ'

compare_integrators([integ_rbf], plot=True, 
                    xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], problem=problem_simple_Gaussian, 
                    dimensions=[0, 1])

# compare_integrators([integ_rbf], plot=True, lim=[-1.0, 1.0], problem=problem_infinite_Gaussian)

# G_rbf, _, containers_rbf, contributions_rbf, stds = integ_rbf(problem, return_std=True, return_all=True)
# print(f"total standard deviation of RBF = {sum(stds)}")
# print(f"maximum standard deviation of RBF = {np.max(stds)}")