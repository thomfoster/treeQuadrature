from treeQuadrature.exampleProblems import PyramidProblem, SimpleGaussian, Camel
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral, SmcIntegral, AdaptiveRbfIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature import compare_integrators

## profiling
from line_profiler import LineProfiler
from treeQuadrature import gaussianProcess
# from treeQuadrature.integrators import gpTreeIntegrator
from treeQuadrature.integrators import TreeIntegrator

### Set problem
# problem = PyramidProblem(D=5)
problem = Camel(D=8)
# problem = SimpleGaussian(D=1)

### set basic parameters
n_samples = 20
max_redraw = 4

N = 13_000
P = 50

### Set ContainerIntegral
rbfIntegral_mean = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, n_samples=n_samples, 
                          range=1e3)
rbfIntegral = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, 
                            fit_residuals=False, 
                            n_samples=n_samples, 
                            range=1e3)
aRbf = AdaptiveRbfIntegral(min_n_samples= int(n_samples / 2), 
                           max_n_samples=int(n_samples * max_redraw), 
                           sample_scaling='exponential')
rmeanIntegral = SmcIntegral(n=n_samples)

split = MinSseSplit()

integ1 = SimpleIntegrator(N, P, split, rbfIntegral_mean)
integ1.name = 'TQ with RBF, fitting to mean'

integ2 = LimitedSampleIntegrator(2000, 500, 10, split, rmeanIntegral, 
                                 lambda container: container.volume)
integ2.name = 'LimitedSampleIntegrator'

integ3 = SimpleIntegrator(N, P, split, rbfIntegral)
integ3.name = 'TQ with RBF'

integ4 = SimpleIntegrator(N, P, split, aRbf)
integ4.name = 'TQ with Adaptive RBF'

integ5 = GpTreeIntegrator(N, P, split, rbfIntegral, grid_size=0.01)
integ5.name = 'Batch GP'

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
    compare_integrators([integ4, integ2], plot=False, 
                        xlim=[0.0, 1.0], ylim=[0.0, 1.0], 
                        problem=problem, dimensions=[0, 1], 
                        n_repeat=5)
    # profiler.disable_by_count()
    # profiler.print_stats()