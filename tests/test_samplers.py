import pytest
import treeQuadrature as tq


samplers = [tq.samplers.UniformSampler(), 
            tq.samplers.ImportanceSampler(),
            tq.samplers.McmcSampler(), 
            tq.samplers.StratifiedSampler(), 
            tq.samplers.SobolSampler(),
            tq.samplers.AdaptiveImportanceSampler(),
            tq.samplers.LHSImportanceSampler()]

@pytest.mark.parametrize('sampler', samplers)
@pytest.mark.parametrize('D', [1, 2, 5])
def test_sampler(sampler, D):
    problem = tq.example_problems.PyramidProblem(D)

    N = 1000

    # Generate samples
    X, y = sampler.rvs(N, mins=problem.lows, maxs=problem.highs, 
                    f = problem.integrand)

    if 'SobolSampler' in str(sampler):
        # number of samples in LowDiscrepancySampler must be power of 2
        assert X.shape[0] <= N
    else:
        assert X.shape[0] == N

    root = tq.Container(X, y, mins=problem.lows, maxs=problem.highs)
    assert (root.X >= problem.lows).all() and (root.X <= problem.highs).all()