from treeQuadrature.exampleProblems import PyramidProblem, SimpleGaussian, Camel, QuadraticProblem
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator, LimitedSamplesGpIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral, AdaptiveRbfIntegral, PolyIntegral, IterativeRbfIntegral
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.samplers import ImportanceSampler, UniformSampler, McmcSampler
from treeQuadrature import compare_integrators

## profiling
from line_profiler import LineProfiler
from treeQuadrature import gaussianProcess
# from treeQuadrature.integrators import gpTreeIntegrator
from treeQuadrature.integrators import TreeIntegrator

### Set problem
# problem = PyramidProblem(D=5)
problem = Camel(D=2)
# problem = SimpleGaussian(D=1)
# problem = QuadraticProblem(D=2)

### set basic parameters
n_samples = 20
max_redraw = 4
max_n_samples = 15_000

N = 8_000
P = 50

### set GP fitter
# gp = SklearnGPFit(alpha=1e-5)

### Set ContainerIntegral
rbfIntegral_mean = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, 
                               n_samples=n_samples)
rbfIntegral = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, 
                            fit_residuals=False, 
                            n_samples=n_samples)
aRbf = AdaptiveRbfIntegral(min_n_samples= int(n_samples / 2), 
                           max_n_samples=int(n_samples * max_redraw))
iRbf = IterativeRbfIntegral()
rmeanIntegral = RandomIntegral(n=n_samples)
polyIntegral = PolyIntegral(n_samples=n_samples, degrees=[2, 3], max_redraw=0)

### set Sampler
iSampler = ImportanceSampler()
uSampler = UniformSampler()    
mcmcSampler = McmcSampler()

### set split
split = MinSseSplit()

integ1 = SimpleIntegrator(N, P, split, rbfIntegral_mean)
integ1.name = 'TQ with RBF, fitting to mean'

integ_is = SimpleIntegrator(N, P, split, aRbf, sampler=iSampler)
integ_is.name = 'Importance sampler'

integ_unif = SimpleIntegrator(N, P, split, aRbf, sampler=mcmcSampler)
integ_unif.name = 'MCMC sampler'

integ_poly = SimpleIntegrator(N, P, split, polyIntegral, sampler=iSampler)
integ_poly.name = 'TQ with polynomial'

integ2 = LimitedSampleIntegrator(2000, 500, 10, split, rmeanIntegral, 
                                 lambda container: container.volume)
integ2.name = 'LimitedSampleIntegrator'

integ_limitedGp = LimitedSamplesGpIntegrator(base_N=N, P=P, max_n_samples=max_n_samples,
                                            split=split, integral=iRbf)
integ_limitedGp.name = 'Limited RbfIntegrator'

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
    compare_integrators([integ_limitedGp], plot=True, verbose=2,
                        xlim=[0.0, 1.0], ylim=[0.0, 1.0], 
                        problem=problem, dimensions=[0, 1], 
                        n_repeat=1)
    # profiler.disable_by_count()
    # profiler.print_stats()