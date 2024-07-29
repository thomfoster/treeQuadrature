### test the container itself.

from treeQuadrature.exampleProblems import Gaussian
import numpy as np

def test_Gaussian(D):
    # create empty container
    gaussian = Gaussian(D, mu=0.0, Sigma=1/200, low=-1.0, high=1.0)
    assert np.isclose(gaussian.answer, 1) 

test_Gaussian(5)