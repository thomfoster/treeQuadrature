from ..container import Container
from .fit_gp import IterativeGPFitting, is_poor_fit
from .visualisation import plot_gp

from typing import Callable
from sklearn.metrics import mean_squared_error


def default_criterion(container: Container) -> bool:
    return True


def gp_diagnosis(
    igp: IterativeGPFitting,
    container: Container,
    criterion: Callable[[Container], bool] = default_criterion,
    plot: bool = False,
) -> None:
    """
    Check the performance of a Gaussian Process (GP) model.

    Parameters
    ----------
    igp : IterativeGPFitting
        The fitted Gaussian Process model.
    container : Container
        Container object that holds the samples and boundaries.
    criterion : function, Optional
        A function that takes a container and returns a boolean indicating
        whether to plot the posterior mean.
        Default criterion: always True
    plot : bool, optional
        if true, 1D problems will be plotted
        Default: False

    Returns
    -------
    None
    """
    xs = igp.gp.X_train_
    ys = igp.gp.y_train_
    n = xs.shape[0]

    # Make predictions
    y_pred, sigma = igp.gp.predict(xs, return_std=True)

    # Check R-squared and MSE
    score = igp.scoring(ys, y_pred, sigma)
    mse = mean_squared_error(ys, y_pred)

    if is_poor_fit(score, igp.performance_threshold, igp.threshold_direction):
        print(f"number of training samples : {n}")
        print(f"volume of container : {container.volume}")
        print(f"GP Score: {score:.3f}")
        print(f"Mean Squared Error: {mse:.3f}")

    # posterior mean plot
    if xs.shape[1] == 1 and criterion(container) and plot:
        plot_gp(igp.gp, xs, ys, mins=container.mins, maxs=container.maxs)
