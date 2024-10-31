from treeQuadrature.example_problems import Ripple, Camel, QuadCamel, ProductPeak
from treeQuadrature.samplers import Sampler, ImportanceSampler, McmcSampler, SobolSampler, StratifiedSampler, AdaptiveImportanceSampler, LHSImportanceSampler
from treeQuadrature import Container
from treeQuadrature.visualisation import plot_containers, plot_integrand

problem = Ripple(D=2)

iSampler = ImportanceSampler()
mcmcSampler = McmcSampler()
mcmc_heated = McmcSampler(temperature=2)
sobolSampler = SobolSampler()
stratifiedSampler = StratifiedSampler()
aiSampler = AdaptiveImportanceSampler()
lhsSampler = LHSImportanceSampler()

def test_sampler(sampler: Sampler, N: int):
    X, y = sampler.rvs(N, mins=problem.lows, maxs=problem.highs, 
                    f = problem.integrand)

    if 'SobolSampler' in str(sampler):
        # number of samples in LowDiscrepancySampler must be power of 2
        assert X.shape[0] <= N
    else:
        assert X.shape[0] == N

    root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    plot_integrand(problem.integrand, 2, xlim=[problem.lows[0], problem.highs[0]],
                   ylim=[problem.lows[1], problem.highs[1]], plot_type='heat')
    plot_containers([root], [1.0], 
                   xlim=[problem.lows[0], problem.highs[0]], 
                   ylim=[problem.lows[1], problem.highs[1]], plot_samples=True)
    
test_sampler(mcmc_heated, 10_000)