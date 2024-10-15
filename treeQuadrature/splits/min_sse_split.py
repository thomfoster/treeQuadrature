import numpy as np
import warnings
from typing import Optional, List

from .base_class import Split
from ..container import Container


class MinSseSplit(Split):
    """
    Partition into two sub-containers
    that minimises variance of f over each set.

    Attributes
    ----------
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be
        in each resulting leaf. \n
        This prevents creating very small partitions
        that might not generalize well.

    scoring_function : callable, optional (default=None)
        A custom function to evaluate the quality of potential splits.

        This function should accept the following parameters:
        the sum of squares of the target values
        on both sides of the split,
        and the number of samples on each side. \n
        It should return a numerical score
        representing the quality of the split.

        - **Parameters:**
            - **sum_left (float):**
                Sum of the target values in the left split.
            - **sum_right (float):**
                Sum of the target values in the right split.
            - **sum_sq_left (float):**
                Sum of the squares of the target values in the left split.
            - **sum_sq_right (float):**
                Sum of the squares of the target values in the right split.
            - **count_left (int):**
                Number of samples in the left split.
            - **count_right (int):**
                Number of samples in the right split.

        - **Returns:**
            - **score (float):** A numerical value representing
              the split quality, where a lower
              score indicates a better split.

        If None, the default relative sum of squared errors
        (RelSSE) is used.

    random_selection: bool, optional (default=False)
        If true, randomly select dimensions to search for split
        instead of using all dimensions.
    """

    def __init__(
        self,
        min_samples_leaf: int = 1,
        scoring_function: Optional[callable] = None,
        random_selection: bool = False
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.scoring_function = scoring_function or relative_sse_score
        self.random_selection = random_selection

    def split(self, container: Container) -> List[Container]:
        """
        Split the container into two sub-containers that minimises
        the sum of squared errors (SSE) or another
        user-defined scoring function.

        Parameters
        ----------
        container : Container
            The container holding the samples and target values.

        Returns
        -------
        List[Container]
            A list containing two sub-containers
            resulting from the best split found.
        """
        samples = container.X
        dims = samples.shape[1]

        ys = container.y.reshape(-1)

        # split wider sides with higher probability. 
        if self.random_selection:
            dimension_weights = container.maxs - container.mins
            dimension_probs = dimension_weights / sum(dimension_weights)

            # Select a proportion of dimensions based on dimension_proportion
            num_selected_dimensions = int(np.sqrt(dims))
            selected_dimensions = np.random.choice(
                range(dims),
                size=num_selected_dimensions,
                p=dimension_probs,
                replace=False
            )
        else:
            selected_dimensions = range(dims)

        best_dimension = -1
        best_thresh = np.inf
        best_score = np.inf

        # Evaluate splits
        for dim in selected_dimensions:
            thresh, score = self.evaluate_split(samples,
                                                ys, dim,
                                                self.min_samples_leaf)
            if score < best_score:
                best_dimension = dim
                best_thresh = thresh
                best_score = score

        if best_thresh == np.inf:  # no split found
            warnings.warn("no split found",
                          RuntimeWarning)
            return [container]

        lcont, rcont = container.split(best_dimension, best_thresh)

        if np.any(lcont.mins == lcont.maxs) or (
            np.any(rcont.mins == rcont.maxs)
        ):
            warnings.warn(
                "Split resulted in a zero-volume container; "
                "reverting to original container"
            )
            return [container]

        return [lcont, rcont]

    def evaluate_split(
        self, samples: np.ndarray, ys: np.ndarray,
        dim: int, min_samples_leaf: int
    ):
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

        return self.findMinSplit(xss, yss, min_samples_leaf)

    def findMinSplit(self, xs: np.ndarray, ys: np.ndarray,
                     min_samples_leaf: int):
        """
        Partition xs and ys such that the custom scoring
        function is minimised.

        Arguments
        --------
        xs, ys: numpy.ndarray
            both 1D array, xs is sorted, ys aligned with xs
        min_samples_leaf: int
            Minimum number of samples required to be in each leaf.
        """

        best_thresh = np.inf
        best_score = np.inf

        n = ys.shape[0]
        sum_left = np.sum(ys[: (min_samples_leaf - 1)])
        sum_right = np.sum(ys) - sum_left
        sum_sq_left = np.sum(ys[: (min_samples_leaf - 1)] ** 2)
        sum_sq_right = np.sum(ys**2) - sum_sq_left

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
                score = self.scoring_function(
                    sum_left,
                    sum_right,
                    sum_sq_left,
                    sum_sq_right,
                    count_left,
                    count_right,
                )

                if score < best_score:
                    best_thresh = xs[i - 1]
                    best_score = score

        return best_thresh, best_score
    

def sse_score(
        sum_left, sum_right, sum_sq_left, sum_sq_right, count_left, count_right
    ):
        """
        Default scoring function: Sum of squared errors (SSE).
        """
        var_left = (sum_sq_left - sum_left**2 / count_left)
        var_right = (sum_sq_right - sum_right**2 / count_right)
        return var_left * count_left + var_right * count_right

def relative_sse_score(
        sum_left, sum_right, sum_sq_left, sum_sq_right, count_left, count_right
    ):
    """
    Divide the variance by sum of squared to remove effect of magnitude. 
    """
    var_left = sum_sq_left - sum_left**2 / count_left
    var_right = sum_sq_right - sum_right**2 / count_right
    return var_left * count_left / sum_sq_left + \
        var_right * count_right / sum_sq_right
