import numpy as np
from .split import Split

class MinSseSplit(Split):
    '''
    Partition into two sub-containers
      that minimize variance of f over each set
    '''
    def split(self, container):
        samples = container.X
        dims = samples.shape[1]

        ys = container.y.reshape(-1)

        best_dimension = -1
        best_thresh = np.inf
        best_score = np.inf

        # Find best split for each dimension and take best
        for dim in range(dims):
            # sort the samples
            xs = np.array(samples[:, dim], copy=True)
            if xs.shape[0] < 2:
                raise RuntimeError(
                    'number of samples in the container not enough for splitting'
                )
            indices = np.argsort(xs)
            xss = xs[indices]
            yss = np.array(ys[indices], copy=True)

            thresh, score = self.findMinSplit(xss, yss)

            if score < best_score:
                best_dimension = dim
                best_thresh = thresh
                best_score = score

        lcont, rcont = container.split(best_dimension, best_thresh)

        return [lcont, rcont]

    @staticmethod
    def defaultObjective(left_y, right_y):
        """
        Comput SSE Impurity of this split

        left_y, right_y : list
            evaluations in left and right sub-containers
        """

        lvar = np.var(left_y)
        rvar = np.var(right_y)

        return lvar * len(left_y) + rvar * len(right_y)

    def findMinSplit(self, xs, ys):
        '''
        Partition xs and ys such that variance across ys subsets is minimized

        Arguments
        --------
        xs, ys: numpy.ndarray
            both 1D array, xs is sorted, ys aligned with xs
        '''

        best_thresh = np.inf
        best_score = np.inf

        threshes = []
        scores = []

        # Iterate through all possible splits :(
        for i in range(1, ys.shape[0]):
            thresh = xs[i - 1]
            score = MinSseSplit.defaultObjective(ys[:i], ys[i:])

            threshes.append(thresh)
            scores.append(score)

            if score < best_score:
                best_thresh = thresh
                best_score = score

        return best_thresh, best_score
