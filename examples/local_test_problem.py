from treeQuadrature.exampleProblems import RippleProblem, SimpleGaussian, Camel, QuadraticProblem, C0Problem, OscillatoryProblem, CornerPeakProblem, ProductPeakProblem, ExponentialProductProblem
from treeQuadrature.integrators import SimpleIntegrator, LimitedSampleIntegrator, GpTreeIntegrator, LimitedSamplesGpIntegrator, SmcIntegrator, DistributedSampleIntegrator, VegasIntegrator
from treeQuadrature.containerIntegration import RandomIntegral, RbfIntegral, AdaptiveRbfIntegral, PolyIntegral, IterativeRbfIntegral
from treeQuadrature.splits import MinSseSplit, KdSplit
from treeQuadrature.samplers import ImportanceSampler, UniformSampler, McmcSampler, SobolSampler, LHSImportanceSampler
from treeQuadrature import compare_integrators

import numpy as np

D = 5

### Set problem
# problem = Camel(D=D)
# problem = SimpleGaussian(D=D)
# problem = RippleProblem(D=D)
# problem = QuadraticProblem(D=D)
# problem = OscillatoryProblem(D, a=np.array(10 / np.linspace(1, D, D)))
# problem = C0Problem(D, np.array([1.1] * D))
# problem = CornerPeakProblem(D=D, a=np.array([10]*D))
# problem = ProductPeakProblem(D=D, a=np.array([10]*D))
problem = ExponentialProductProblem(D)

### set basic parameters
n_samples = 30
max_redraw = 4
max_n_samples = int(15_000 * (D/3))

N = int(10_000 * (D/3))
P = 50

### set GP fitter
# gp = SklearnGPFit(alpha=1e-5)

### Set ContainerIntegral
rbfIntegral_mean = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, 
                               n_samples=n_samples)
rbfIntegral = RbfIntegral(max_redraw=max_redraw, threshold=0.5, n_splits=3, 
                            fit_residuals=False, 
                            n_samples=n_samples)
rbfIntegral_non_iter = RbfIntegral(max_redraw=0, n_splits=0, 
                                   n_samples=n_samples)
aRbf = AdaptiveRbfIntegral(n_samples= n_samples, max_redraw=0, n_splits=0)
iRbf = IterativeRbfIntegral(n_samples=n_samples, n_splits=0)
rmeanIntegral = RandomIntegral(n_samples=n_samples)
polyIntegral = PolyIntegral(n_samples=n_samples, degrees=[2, 3], max_redraw=0)

### set Sampler
iSampler = ImportanceSampler()
uSampler = UniformSampler()    
mcmcSampler = McmcSampler()
sobolSampler = SobolSampler()
lhsSampler = LHSImportanceSampler()

### set split
split = MinSseSplit()
# split = KdSplit()

integ1 = SimpleIntegrator(N, P, split, rbfIntegral_mean)
integ1.name = 'TQ with RBF, fitting to mean'

integ_mean = DistributedSampleIntegrator(N, P, max_n_samples, split, rmeanIntegral, sampler=mcmcSampler)
integ_mean.name = 'TQ with mean estimator'

integ_is = SimpleIntegrator(N, P, split, aRbf, sampler=iSampler)
integ_is.name = 'Importance sampler'

integ_unif = SimpleIntegrator(N, P, split, aRbf, sampler=mcmcSampler)
integ_unif.name = 'MCMC sampler'

integ_poly = SimpleIntegrator(N, P, split, polyIntegral, sampler=iSampler)
integ_poly.name = 'TQ with polynomial'

lsi_N = int(max_n_samples / (n_samples + 1))
integ_active = LimitedSampleIntegrator(lsi_N, 500, 10, split, rmeanIntegral, 
                                 lambda container: container.volume, sampler=mcmcSampler)
integ_activeTQ = DistributedSampleIntegrator(N, P, max_n_samples, split, rmeanIntegral, sampler=mcmcSampler,
                                             construct_tree_method=integ_active.construct_tree)
integ_activeTQ.name = 'LimitedSampleIntegrator'

integ_limitedGp = LimitedSamplesGpIntegrator(base_N=N, P=P, max_n_samples=max_n_samples,
                                            split=split, integral=iRbf, sampler=lhsSampler, 
                                            max_container_samples=100)
integ_limitedGp.name = 'Limited RbfIntegrator'

integ_rbf = DistributedSampleIntegrator(N, P, max_n_samples, split, aRbf, sampler=lhsSampler, 
                                        min_container_samples=10)
integ_rbf.name = 'TQ with RBF'

integ4 = SimpleIntegrator(N, P, split, aRbf)
integ4.name = 'TQ with Adaptive RBF'

integ_batch = GpTreeIntegrator(N, P, split, rbfIntegral_non_iter, 
                               sampler = lhsSampler, max_n_samples=max_n_samples)
integ_batch.name = 'Batch GP'

integ_smc = SmcIntegrator(N=max_n_samples, sampler=UniformSampler())
integ_smc.name = 'SMC'

n_iter = 10
adaptive_iter = 0
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas = VegasIntegrator(vegas_n, n_iter, adaptive_iter)
integ_vegas.name = 'Vegas'

adaptive_iter = 5
vegas_n = int(max_n_samples / (n_iter + adaptive_iter))
integ_vegas_adaptive = VegasIntegrator(vegas_n, n_iter, adaptive_iter)
integ_vegas_adaptive.name = 'Adaptive Vegas'

if __name__ == '__main__':
    compare_integrators([integ_vegas, integ_vegas_adaptive], plot=True, verbose=1,
                        xlim=[problem.lows[0], problem.highs[0]], 
                        ylim=[problem.lows[1], problem.highs[1]], 
                        problem=problem, dimensions=[0, 1], 
                        n_repeat=3, integrator_specific_kwargs={
                            'LimitedSampleIntegrator': {'integrand' : problem.integrand}})