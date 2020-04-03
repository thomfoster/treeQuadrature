import pytest
import treeQuadrature as tq
from functools import partial

@pytest.mark.parametrize("D", [1,2])
@pytest.mark.parametrize("n_samples", [0,1,10])
@pytest.mark.parametrize("distribution", [
    partial(tq.exampleDistributions.Uniform, low=-1.0, high=1.0),
    tq.exampleDistributions.Camel,
    tq.exampleDistributions.QuadCamel
])
def test_shapes(D, n_samples, distribution):
    d = distribution(D)

    X = d.rvs(n_samples)
    assert X.ndim == 2
    assert X.shape[0] == n_samples
    assert X.shape[1] == D

    y = d.pdf(X)
    assert y.ndim == 2
    assert y.shape[0] == n_samples
    assert y.shape[1] == 1