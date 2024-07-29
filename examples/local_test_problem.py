from treeQuadrature.exampleProblems import PyramidProblem, SimpleGaussian, Camel
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature import compare_integrators
import numpy as np

## profiling
from line_profiler import LineProfiler
from treeQuadrature import gaussianProcess
# from treeQuadrature.integrators import gpTreeIntegrator
from treeQuadrature.integrators import TreeIntegrator

# problem = PyramidProblem(D=5)
# problem = Camel(D=2)
problem = SimpleGaussian(D=1)

rbfIntegral = RbfIntegral(max_redraw=4, threshold=0.5, n_splits=3)
rbfIntegral_2 = RbfIntegral(range=30, max_redraw=4, threshold=0.5, n_splits=0)
rmedianIntegral = RandomIntegral(n=20)
rmeanIntegral = SmcIntegral(n=20)

split = MinSseSplit()

integ1 = SimpleIntegrator(8_000, 50, split, rbfIntegral)
integ1.name = 'TQ with RBF'

integ2 = GpTreeIntegrator(8_000, 50, split, rbfIntegral_2, grid_size=0.01)
integ2.name = 'TQ with RBF and hyper-parameter passing'

integ3 = LimitedSampleIntegrator(1000, 500, 10, split, rmeanIntegral, 
                                 lambda container: container.volume)
integ3.name = 'LimitedSampleIntegrator'


if __name__ == '__main__':
    ### profile to test runnning time
    profiler = LineProfiler()
    profiler.add_function(TreeIntegrator.__call__)
    # profiler.add_function(gaussianProcess.gp_kfoldCV)
    profiler.add_function(gaussianProcess.IterativeGPFitting.fit)
    # profiler.add_function(GpTreeIntegrator.__call__)
    # profiler.add_function(GpTreeIntegrator.fit_gps)
    # profiler.add_function(gpTreeIntegrator.build_grid)
    # profiler.add_function(gpTreeIntegrator.find_neighbors_grid)


    # profiler.enable_by_count()
    compare_integrators([integ3], plot=False, 
                        xlim=[0.0, 1.0], ylim=[0.0, 1.0], 
                        problem=problem, verbose=True, dimensions=[0, 1])
    # profiler.disable_by_count()
    # profiler.print_stats()