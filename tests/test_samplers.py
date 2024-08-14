import pytest
import treeQuadrature as tq


samplers = [tq.samplers.UniformSampler(), 
            tq.samplers.ImportanceSampler(),
            tq.samplers.McmcSampler(), 
            tq.samplers.StratifiedSampler(strata_per_dim=2)]

@pytest.mark.parametrize('sampler', samplers)
@pytest.mark.parametrize('D', [1, 2, 5])
def test_sampler(sampler, D):
    problem = tq.exampleProblems.PyramidProblem(D)

    N = 1000

    # Generate samples
    X = sampler.rvs(N, problem)
    assert X.shape[0] == N

    # Evaluate the integrand
    y = problem.integrand(X)
    root = tq.Container(X, y, mins=problem.lows, maxs=problem.highs)
    assert root.X.shape[0] == N
    assert root.y.shape[0] == N
    assert (root.X >= problem.lows).all() and (root.X <= problem.highs).all()