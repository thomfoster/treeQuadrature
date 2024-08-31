import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from .fit_gp import GPFit
from ..container import Container


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
        _plot_gp_1d(gp, xs, ys, mins[0], maxs[0], plot_ci)
    elif xs.shape[1] == 2:
        assert (
            len(mins) == 2 and len(maxs) == 2
        ), (
            "mins and maxs must have two elements "
            "for 2-dimensional problems")
        _plot_gp_2d(gp, xs, ys, mins[0], maxs[0],
                    mins[1], maxs[1], plot_ci)
    else:
        raise ValueError(
            "This function only supports 1-dimensional "
            "and 2-dimensional problems"
        )


def _plot_gp_1d(
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


def _plot_gp_2d(
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
    Plot the Gaussian Process posterior mean
    and the data points for a 2D problem.

    Parameters
    ----------
    gp : GPFit
        The trained GP model.
    xs, ys : numpy.ndarray
        The data points.
    x_min, x_max : float
        The lower and upper bounds for the
        first axis (x-axis) for plotting.
    y_min, y_max : float
        The lower and upper bounds for the
        second axis (y-axis) for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. \n
          Default is True.
    """
    assert xs.shape[1] == 2, (
        "This function only supports 2-dimensional problems"
    )

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


default_title = 'GP Posterior Mean Functions within Containers'


def plot_gps(gp_fits: List[GPFit],
             containers: List[Container],
             colormap=cm.plasma,
             title: str = default_title,
             alpha: float = 0.8,
             sample_color='r',
             plot_samples=True,
             grid_points=100):
    """
    Plot the Gaussian Process (GP) posterior mean function for
    each GPFit instance over the given containers.

    Arguments
    ---------
    gp_fits : List[GPFit]
        The fitted GP models.
    containers : list
        A list of containers,
        each with 'mins' and 'maxs' attributes
        defining the integration domain.
    grid_points : int, optional
        The number of points to generate for
        the grid in each dimension. \n
        Default: 100.
    title : str, optional
        The title of the plot. \n
        Default: 'GP Posterior Mean Functions within Containers'. \n
        Set to None to disable the title.
    alpha : float
        The transparency of the surfaces. \n
        Default: 0.8. \n
        Only used for 2D problems.
    plot_samples : bool, optional
        If True, the sample points will be plotted. \n
        Default: True.
    sample_color : str, optional
        The color of the sample points. \n
        Default: 'r'. (red) \n
        Only used for 2D problems.
    colormap : colormap instance, optional
        The colormap to use for coloring the surfaces (default: cm.plasma).
    """
    num_gps = len(gp_fits)
    num_containers = len(containers)

    if num_gps != num_containers:
        raise ValueError(
            "The number of GPFit instances must "
            "match the number of Containers.")

    # We assume that all Containers have the same dimension D
    D = len(containers[0].mins)

    if D == 1:
        plt.figure(figsize=(8, 6))

        all_y_means = [gp_fit.y_mean for gp_fit in gp_fits]
        min_y_mean = min(all_y_means)
        max_y_mean = max(all_y_means)
        for i, (gp_fit, container) in enumerate(zip(gp_fits, containers)):
            mins = container.mins
            maxs = container.maxs

            # Generate a grid of points for plotting
            x_grid = np.linspace(mins[0], maxs[0], grid_points)
            grid = x_grid.reshape(-1, 1)

            # Predict the GP mean at each grid point
            y_mean = gp_fit.y_mean
            y_pred = gp_fit.predict(grid, return_std=False) + y_mean

            normalized_y_mean = (
                y_mean - min_y_mean) / (max_y_mean - min_y_mean)
            color = colormap(normalized_y_mean)

            # Plot the GP mean curve
            plt.plot(x_grid, y_pred, color=color)

            # Plot fitting points
            y_points = gp_fit.y_train_ + y_mean
            plt.scatter(
                gp_fit.X_train_, y_points,
                color=color, edgecolor='k', zorder=5)

            # Plot vertical lines to mark the boundaries of the container
            y_min_boundary = gp_fit.predict(
                np.array([[mins[0]]]), return_std=False) + y_mean
            y_max_boundary = gp_fit.predict(
                np.array([[maxs[0]]]), return_std=False) + y_mean
            plt.plot([mins[0], mins[0]],
                     [0, y_min_boundary.item()], color='k',
                     linestyle='--', linewidth=1)
            plt.plot([maxs[0], maxs[0]],
                     [0, y_max_boundary.item()], color='k',
                     linestyle='--', linewidth=1)

        plt.xlabel('x')
        plt.ylabel('GP Mean')
        plt.title(title)
        plt.grid(True)
        plt.legend([Line2D([0], [0], color='k',
                    linestyle='--', linewidth=1)],
                   ['Boundary'])
        plt.show()

    elif D == 2:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize colors based on the mean height
        all_means = []
        for gp_fit, container in zip(gp_fits, containers):
            mins = container.mins
            maxs = container.maxs
            x1_grid = np.linspace(mins[0], maxs[0], grid_points)
            x2_grid = np.linspace(mins[1], maxs[1], grid_points)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            grid = np.c_[X1.ravel(), X2.ravel()]
            y_mean = gp_fit.y_mean
            y_pred = gp_fit.predict(grid, return_std=False) + y_mean
            Z = y_pred.reshape(X1.shape)
            mean_height = Z.mean()
            all_means.append(mean_height)

        # Normalize mean heights to [0, 1] for colormap
        min_mean = min(all_means)
        max_mean = max(all_means)
        norm_means = [(m - min_mean) / (max_mean - min_mean)
                      for m in all_means]

        # Plot each GP surface with a color based on its mean height
        for i, (gp_fit, container) in enumerate(zip(gp_fits, containers)):
            mins = container.mins
            maxs = container.maxs

            x1_grid = np.linspace(mins[0], maxs[0], grid_points)
            x2_grid = np.linspace(mins[1], maxs[1], grid_points)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            grid = np.c_[X1.ravel(), X2.ravel()]

            y_mean = gp_fit.y_mean
            y_pred = gp_fit.predict(grid, return_std=False) + y_mean
            Z = y_pred.reshape(X1.shape)

            # Get the color for this surface based on its mean height
            color_value = norm_means[i]
            color = colormap(color_value)

            # Plot the GP surface with the selected color
            facecolors = np.tile(color, (Z.shape[0], Z.shape[1], 1))
            ax.plot_surface(
                X1, X2, Z, facecolors=facecolors, alpha=alpha,
                rstride=grid_points//5, cstride=grid_points//5,
                edgecolor='gray', linewidth=0.5)

            # Plot fitting points (X_train_ and y_train_)
            y_points = gp_fit.predict(gp_fit.X_train_,
                                      return_std=False) + y_mean
            ax.scatter(gp_fit.X_train_[:, 0], gp_fit.X_train_[:, 1],
                       y_points,
                       color=sample_color, edgecolor='k', s=50, zorder=5)

        ax.set_zlabel('GP Mean')
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.xaxis._axinfo["grid"].update(
            color='black', linestyle='-', linewidth=1.5)
        ax.yaxis._axinfo["grid"].update(
            color='black', linestyle='-', linewidth=1.5)
        ax.zaxis._axinfo["grid"].update(
            color='black', linestyle='-', linewidth=1.5)
        ax.set_title(title)
        plt.show()

    else:
        print(f"Plotting for D={D} dimensions is not supported.")
