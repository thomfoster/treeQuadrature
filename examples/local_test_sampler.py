from treeQuadrature.exampleProblems import PyramidProblem, RippleProblem
from treeQuadrature.samplers import ImportanceSampler, McmcSampler, SobolSampler, StratifiedSampler
from treeQuadrature import Container
from treeQuadrature.visualisation import plotContainers

from inspect import signature

problem = RippleProblem(D=2)

iSampler = ImportanceSampler()
mcmcSampler = McmcSampler()
sobolSampler = SobolSampler()
stratifiedSampler = StratifiedSampler()

def test_sampler(sampler, N):
    signatures = signature(sampler.rvs).parameters
    if 'f' in signatures:
        X = sampler.rvs(N, mins=problem.lows, maxs=problem.highs, 
                        f = problem.integrand)
    else: 
        X = sampler.rvs(N, mins=problem.lows, maxs=problem.highs)

    if 'SobolSampler' in str(sampler):
        # number of samples in LowDiscrepancySampler must be power of 2
        assert X.shape[0] <= N
    else:
        assert X.shape[0] == N

    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    plotContainers([root], [1.0], 
                   xlim=[problem.lows[0], problem.highs[0]], 
                   ylim=[problem.lows[1], problem.highs[1]], plot_samples=True)
    
test_sampler(mcmcSampler, 5_000)