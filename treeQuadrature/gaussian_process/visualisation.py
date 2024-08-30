import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .fit_gp import GPFit


def plot_gp(
    gp: GPFit,
    xs: np.ndarray,
    ys: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    plot_ci: Optional[bool] = True,
):
    """
    Plot the Gaussian Process posterior mean and the data points.

    Parameters
    ----------
    gp : GPFit
        The trained GP model.
    xs, ys : numpy.ndarray
        The data points.
    mins : np.ndarray
        The lower bounds for plotting.
    maxs : np.ndarray
        The upper bounds for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. Default is True.
    """
    if xs.shape[1] == 1:
        _plot_GP_1D(gp, xs, ys, mins[0], maxs[0], plot_ci)
    elif xs.shape[1] == 2:
        assert (
            len(mins) == 2 and len(maxs) == 2
        ), "mins and maxs must have two elements for 2-dimensional problems"
        _plot_GP_2D(gp, xs, ys, mins[0], maxs[0], mins[1], maxs[1], plot_ci)
    else:
        raise ValueError(
            "This function only supports 1-dimensional and 2-dimensional problems"
        )


def _plot_GP_1D(
    gp: GPFit,
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    x_max: float,
    plot_ci: Optional[bool] = True,
):
    """
    Plot the Gaussian Process posterior mean
    and the data points

    Parameters
    ----------
    gp : GPFit
        the trained GP model
    xs, ys : numpy.ndarray
        the data points
    x_min, x_max : float
        the lower and upper bounds for plotting
    plot_ci : bool
        if True, the confidence interval will be plotted.
        Default True
    """
    assert xs.shape[1] == 1, "only supports 1-dimensional problems"

    x_plot = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # Predict the mean and standard deviation of the GP model
    y_mean, y_std = gp.predict(x_plot, return_std=True)

    # Plot the original points
    plt.scatter(xs, ys, c="r", marker="x", label="Data points")

    # Plot the GP mean function
    plt.plot(x_plot, y_mean, "b-", label="GP mean")

    # Plot the confidence interval
    if plot_ci:
        plt.fill_between(
            x_plot.ravel(),
            y_mean - 1.96 * y_std,
            y_mean + 1.96 * y_std,
            alpha=0.2,
            color="b",
            label="95% confidence interval",
        )

    # Add labels and legend
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Gaussian Process Regression")
    plt.legend()
    plt.show()


def _plot_GP_2D(
    gp: GPFit,
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_ci: Optional[bool] = True,
):
    """
    Plot the Gaussian Process posterior mean and the data points for a 2D problem.

    Parameters
    ----------
    gp : GPFit
        The trained GP model.
    xs, ys : numpy.ndarray
        The data points.
    x_min, x_max : float
        The lower and upper bounds for the first axis (x-axis) for plotting.
    y_min, y_max : float
        The lower and upper bounds for the second axis (y-axis) for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. Default is True.
    """
    assert xs.shape[1] == 2, "This function only supports 2-dimensional problems"

    # Create a grid over the input space
    x1_plot = np.linspace(x_min, x_max, 100)
    x2_plot = np.linspace(y_min, y_max, 100)
    x1_grid, x2_grid = np.meshgrid(x1_plot, x2_plot)
    x_plot = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    # Predict the mean and standard deviation of the GP model
    y_mean, y_std = gp.predict(x_plot, return_std=True)
    y_mean = y_mean.reshape(x1_grid.shape)
    y_std = y_std.reshape(x1_grid.shape)

    # Plot the GP mean function
    plt.figure(figsize=(12, 6))
    plt.contourf(x1_grid, x2_grid, y_mean, cmap="viridis", alpha=0.8)
    plt.colorbar(label="GP mean")

    # Plot the original points
    scatter = plt.scatter(
        xs[:, 0],
        xs[:, 1],
        c=ys,
        cmap="viridis",
        edgecolors="k",
        marker="o",
        label="Data points",
    )

    # Plot the confidence interval
    if plot_ci:
        ci_lower = y_mean - 1.96 * y_std
        ci_upper = y_mean + 1.96 * y_std
        plt.contour(
            x1_grid,
            x2_grid,
            ci_lower,
            levels=1,
            colors="blue",
            linestyles="dashed",
            alpha=0.5,
        )
        plt.contour(
            x1_grid,
            x2_grid,
            ci_upper,
            levels=1,
            colors="red",
            linestyles="dashed",
            alpha=0.5,
        )

        # Add dummy plots for the legend
        lower_dummy_line = plt.Line2D(
            [0], [0], linestyle="dashed", color="blue", alpha=0.5
        )
        upper_dummy_line = plt.Line2D(
            [0], [0], linestyle="dashed", color="red", alpha=0.5
        )
        plt.legend(
            [scatter, lower_dummy_line, upper_dummy_line],
            ["Data points", "95% CI Lower Bound", "95% CI Upper Bound"],
        )

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Gaussian Process Regression for 2D Problems")
    plt.show()
