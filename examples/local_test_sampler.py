from treeQuadrature.exampleProblems import PyramidProblem
from treeQuadrature.samplers import ImportanceSampler, McmcSampler
from treeQuadrature import Container
from treeQuadrature.visualisation import plotContainers

problem = PyramidProblem(D=2)

iSampler = ImportanceSampler()
mcmcSampler = McmcSampler()

def test_sampler(sampler, N):
    X = sampler.rvs(N, problem)

    assert X.shape[0] == N

    y = problem.integrand(X)
    root = Container(X, y, mins=problem.lows, maxs=problem.highs)

    plotContainers([root], [1.0], 
                   xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], plot_samples=True)
    
test_sampler(mcmcSampler, 2_000)