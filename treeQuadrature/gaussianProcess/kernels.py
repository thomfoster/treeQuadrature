from sklearn.gaussian_process.kernels import Kernel
import numpy as np

class Polynomial(Kernel):
    """Polynomial kernel for Gaussian Processes."""

    def __init__(self, degree=3, coef0=1.0):
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return (np.dot(X, Y.T) + self.coef0) ** self.degree

    def diag(self, X):
        return np.sum(X**2, axis=1)**self.degree + self.coef0

    def is_stationary(self):
        return False

    def clone_with_theta(self, theta):
        return Polynomial(degree=self.degree, coef0=self.coef0)
    
    def __repr__(self):
        return f"PolynomialKernel(degree={self.degree}, coef0={self.coef0})"