from treeQuadrature.exampleProblems import PyramidProblem, SimpleGaussian, Camel
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral, SmcIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature import compare_integrators

## profiling
from line_profiler import LineProfiler
from treeQuadrature import gaussianProcess
# from treeQuadrature.integrators import gpTreeIntegrator
from treeQuadrature.integrators import TreeIntegrator

# problem = PyramidProblem(D=5)
problem = Camel(D=2)
# problem = SimpleGaussian(D=1)

rbfIntegral = RbfIntegral(max_redraw=4, threshold=0.5, n_splits=3)
rmedianIntegral = RandomIntegral(n=20)
rmeanIntegral = SmcIntegral(n=20)

split = MinSseSplit()

integ1 = SimpleIntegrator(8_000, 50, split, rbfIntegral)
integ1.name = 'TQ with RBF, fitting to mean'

integ2 = LimitedSampleIntegrator(1000, 500, 10, split, rmeanIntegral, 
                                 lambda container: container.volume)
integ2.name = 'LimitedSampleIntegrator'

rbfIntegral_2 = RbfIntegral(max_redraw=4, threshold=0.5, n_splits=3, fit_residuals=False)
integ3 = SimpleIntegrator(8_000, 50, split, rbfIntegral_2)
integ3.name = 'TQ with RBF'

integ4 = GpTreeIntegrator(8000, 40, split, rbfIntegral, grid_size=0.01)


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
    compare_integrators([integ4, integ1], plot=False, 
                        xlim=[0.0, 1.0], ylim=[0.0, 1.0], 
                        problem=problem, verbose=False, dimensions=[0, 1], 
                        n_repeat=1)
    # profiler.disable_by_count()
    # profiler.print_stats()