import numpy as np
import warnings

from .split import Split
from ..container import Container

class MinSseSplit(Split):
    '''
    Partition into two sub-containers
      that minimises variance of f over each set

    Attribute
    ---------
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be in each resulting leaf.
        This prevents creating very small partitions that might not generalize well.
    '''
    def __init__(self, min_samples_leaf: int=1) -> None:
        self.min_samples_leaf = min_samples_leaf

    def split(self, container: Container):
        """
        Split the container into two sub-containers that minimises
        the sum of squared errors (SSE) of the target values in each set.

        Parameters
        ----------
        container : Container
            The container holding the samples and target values.
    
        Returns
        -------
        List[Container]
            A list containing two sub-containers resulting from the best split found.
        """
        samples = container.X
        dims = samples.shape[1]

        ys = container.y.reshape(-1)

        best_dimension = -1
        best_thresh = np.inf
        best_score = np.inf

        # Evaluate splits
        for dim in range(dims):
            thresh, score = self.evaluate_split(samples, ys, dim, 
                                                self.min_samples_leaf)
            if score < best_score:
                best_dimension = dim
                best_thresh = thresh
                best_score = score

        if best_thresh == np.inf: # no split found
            warnings.warn('no split found')
            return [container]
        
        lcont, rcont = container.split(best_dimension, best_thresh)

        return [lcont, rcont]

    @staticmethod
    def evaluate_split(samples: np.ndarray, ys: np.ndarray, 
                       dim: int, min_samples_leaf: int):
        """
        Evaluate the best split for a given dimension.

        Parameters
        ----------
        samples : numpy.ndarray
            The sample points.
        ys : numpy.ndarray
            The target values.
        dim : int
            The dimension along which to split.
        min_samples_leaf : int
            Minimum number of samples required in each leaf.

        Returns
        -------
        tuple
            A tuple containing the best threshold and the best score.
        """
        # sort the samples
        xs = np.array(samples[:, dim], copy=True)
        if xs.shape[0] < 2:
            return np.inf, np.inf

        indices = np.argsort(xs)
        xss = xs[indices]
        yss = np.array(ys[indices], copy=True)

        return MinSseSplit.findMinSplit(xss, yss, min_samples_leaf)

    @staticmethod
    def findMinSplit(xs: np.ndarray, ys: np.ndarray, 
                     min_samples_leaf: int):
        '''
        Partition xs and ys such that variance across ys subsets is minimized

        Arguments
        --------
        xs, ys: numpy.ndarray
            both 1D array, xs is sorted, ys aligned with xs
        min_samples_leaf: int
            Minimum number of samples required to be in each leaf.
        '''

        best_thresh = np.inf
        best_score = np.inf

        n = ys.shape[0]
        sum_left = 0.0
        sum_right = np.sum(ys)
        sum_sq_left = 0.0
        sum_sq_right = np.sum(ys ** 2)

        # Iterate through all possible splits
        for i in range(min_samples_leaf, n - min_samples_leaf + 1):
            sum_left += ys[i - 1]
            sum_right -= ys[i - 1]
            sum_sq_left += ys[i - 1] ** 2
            sum_sq_right -= ys[i - 1] ** 2

            if i < n and xs[i] == xs[i - 1]:
                continue  # Skip splits that are not actually splits

            count_left = i
            count_right = n - i

            if count_left > 0 and count_right > 0:
                var_left = (sum_sq_left - (sum_left ** 2) / count_left) / count_left
                var_right = (sum_sq_right - (sum_right ** 2) / count_right) / count_right
                score = var_left * count_left + var_right * count_right

                if score < best_score:
                    best_thresh = xs[i - 1]
                    best_score = score

        return best_thresh, best_score
