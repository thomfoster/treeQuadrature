import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.axes import Axes
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from typing import Optional, List, Callable

from .utils import scale
from .container import Container

def plotContainers(containers: List[Container], contributions: List[float], 
                   xlim: List[float], ylim: Optional[List[float]]=None, 
                   integrand: Optional[Callable]=None, 
                   title: Optional[str]=None, 
                   plot_samples: Optional[bool]=False, 
                   dimensions: Optional[List[float]]=None):
    """
    Plot containers and their contributions
    for 1D problems, the integrand can be plotted

    Parameters
    ----------
    containers : list
        list of Container objects
    contributions : list
        numerical values of contributions of each container
    xlim : list of 2 floats
        the range of x-axis
    ylim : list of 2 floats, optional
        the range of y-axis
        ignored by 1D problems
    integrand : function, optional
        ignored by 2D problems
    title : String, optional
        title of the plot
    plot_samples : bool, optional
        if True, samples will be placed on the plot as well. 
        Defaults to False
    dimensions : list of int, optional
        The two dimensions to plot. 
        If None, plot all dimensions.        
    """

    assert len(containers) == len(contributions), (
        'The length of containers and contributions must be the same'
        f'got {len(containers)} and {len(contributions)}'
    )

    # check dimensions
    all_dimensions = list(range(containers[0].D))
    if dimensions is None:
        dimensions = all_dimensions
    elif len(dimensions) != 2:
        raise ValueError('dimensions must have length 2')
    
    assert (dimensions[0] in all_dimensions) and (
        dimensions[1] in all_dimensions), (
            f'dimensions must be in 0, 1, ..., {len(all_dimensions)-1}'
        )

    if containers[0].D ==1:
        _plotContainers1D(containers, contributions, xlim, 
                          integrand, title, plot_samples)
    elif len(dimensions) == 2:
        _plotContainers2D(containers, contributions, xlim, ylim, title, 
                          plot_samples, dimensions[0], dimensions[1])
    else:
        raise ValueError(
            "Only 1D and 2D plots are supported. "
             "Please provide 2 dimensions to plot for higher dimensions"
             )


def _plotContainers2D(containers, contributions, xlim, ylim, title, 
                      plot_samples, dim1, dim2):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = colormaps['YlOrRd'].resampled(256)

    if len(contributions) > 1:
        contributions = scale(contributions)

    for container, contribution in zip(containers, contributions):
        plotContainer(ax, container, dim1, dim2, 
                      plot_samples=plot_samples, facecolor=cmap(contribution), alpha=0.4)

    if title:
        plt.title(title)
    else:
        plt.title('Container Contributions')
    plt.show()

def _plotContainers1D(containers, contributions, xlim, integrand, title, plot_samples):
    if integrand is not None:
        ### plot the integrand
        x_values = np.linspace(xlim[0], xlim[1], 2000).reshape(-1, 1)
        y_values = integrand(x_values)
        plt.plot(x_values, y_values, label='Integrand', color='blue')
    
    ### Plot the contributions as a bar chart
    container_centers = []
    container_widths = []
    
    for container in containers:
        center = (container.mins[0] + container.maxs[0]) / 2
        width = container.maxs[0] - container.mins[0]
        container_centers.append(center)
        container_widths.append(width)
    
    # scale contributions by container volume
    volumes = np.array([container.volume for container in containers])
    plt.bar(container_centers, contributions / volumes, width=container_widths, 
            alpha=0.5, color='red', label='Contributions')
    
    ### plot samples
    if plot_samples:
        all_samples = np.concatenate([container.X for container in containers if container.X.size > 0])
        plt.scatter(all_samples, np.full_like(all_samples, 0.05), color='red', marker='x', label='Samples')
    
    ### Setting the canvas
    plt.xlim(xlim)
    # labels and legend
    plt.xlabel('x')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    else:
        plt.title('Integrand and Container Contributions')
    plt.legend()

    plt.show()

def plotContainer(ax: Axes, container: Container, dim1: int, dim2: int, 
                  **kwargs):
    '''
    Plot a container on the provided axes.

    Parameters
    --------
    ax: Axes
        the canvas
    container : Container
    dim1 : int
        the first dimension to plot
    dim2 : int
        the second dimension to plot
    **kwargs
        keyword arguments, can be passed to matplotlib.patches.Rectangle
        plot_samples : bool
            if true, produce scatter plot of samples
            Defaulst to True
        s : float
            marker size of scatter plot. 
            Only valid when plot_samples = True
            Defaults to 5.0
        fill : bool
            if true, fill the containers with colours
            Defaults to True
        ec : color
            edge colour
            Defaults to black
    '''

    # kwargs
    plot_samples = kwargs.pop('plot_samples', True)
    s = kwargs.pop('s', 5.0)
    fill = kwargs.pop('fill', True)
    ec = kwargs.pop('ec', 'black')

    # Plotting container samples
    if plot_samples:
        ax.scatter(container.X[:, dim1], container.X[:, dim2],
                   color='navy', s=s, alpha=0.3)

    # Plot container boundary
    x1, y1 = container.mins[dim1], container.mins[dim2]
    x2, y2 = container.maxs[dim1], container.maxs[dim2]
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=fill,
        ec=ec,
        **kwargs
    )
    ax.add_patch(rect)

def plotIntegrand(integrand, D, xlim, ylim=None, n_points=500, levels=10):
    """
    Plot the integrand
    2D problems: Contour lines used

    Parameters
    ----------
    integrand : function
        the function to be plotted
    D : int
        dimension of the input space
    xlim : list of 2 floats
        the range of x-axis
    ylim : list of 2 floats, optional
        the range of y-axis
        ignored by 1D problems
    n_points: int, optioanl
        number of points in each dimension for 
        plotting the Contours
        Defaults to 500
    levels : int, optional
        levels of the Contour plot
        ignored when D = 1
        Defaults to 10
    """

    if D == 1:
        _plot1D(integrand, xlim, n_points)
    elif D == 2:
        _plot2D(integrand, xlim, ylim, levels, n_points)
    else:
        raise Exception('only supports 1D and 2D problems')

def _plot1D(f, xlim, n_points):
    x = np.linspace(xlim[0], xlim[1], n_points)
    ys = f(x)

    plt.plot(x, ys)
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.show()

def _plot2D(f, xlim, ylim, levels, n_points):
    """
    plot Contour lines of a 2D distribution 
    Automatically adapts the grid to significant values
    """

    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([xs, ys])[0,0] for xs in x] for ys in y])

    ## adapt the window to PDF
    threshold = 1e-3
    mask = Z > threshold

    # Find indices of the bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Trim the X, Y, Z arrays
    X_trimmed = X[min_row:max_row+1, min_col:max_col+1]
    Y_trimmed = Y[min_row:max_row+1, min_col:max_col+1]
    Z_trimmed = Z[min_row:max_row+1, min_col:max_col+1]

    contour = plt.contour(X_trimmed, Y_trimmed, Z_trimmed, levels=levels, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(contour)
    plt.show()


def plotGP(gp: GaussianProcessRegressor, xs: np.ndarray, ys: np.ndarray, 
           mins: np.ndarray, maxs: np.ndarray, plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean and the data points.

    Parameters
    ----------
    gp : sklearn.gaussian.process.GaussianProcessRegressor
        The trained GP regressor.
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
        _plotGP1D(gp, xs, ys, mins[0], maxs[0], plot_ci)
    elif xs.shape[1] == 2:
        assert len(mins) == 2 and len(maxs) == 2, (
            'mins and maxs must have two elements for 2-dimensional problems'
            )
        _plotGP2D(gp, xs, ys, mins[0], maxs[0], mins[1], maxs[1], plot_ci)
    else:
        raise ValueError('This function only supports 1-dimensional and 2-dimensional problems')

def _plotGP1D(gp: GaussianProcessRegressor, xs: np.ndarray, ys: np.ndarray, 
           x_min: float, x_max: float, plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean
    and the data points

    Parameters
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        the trained GP regressor
    xs, ys : numpy.ndarray
        the data points
    x_min, x_max : float
        the lower and upper bounds for plotting
    plot_ci : bool
        if True, the confidence interval will be plotted.
        Default True
    """
    assert xs.shape[1] == 1, 'only supports 1-dimensional problems'

    x_plot = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # Predict the mean and standard deviation of the GP model
    y_mean, y_std = gp.predict(x_plot, return_std=True)

    # Plot the original points
    plt.scatter(xs, ys, c='r', marker='x', label='Data points')

    # Plot the GP mean function
    plt.plot(x_plot, y_mean, 'b-', label='GP mean')

    # Plot the confidence interval
    if plot_ci:
        plt.fill_between(x_plot.ravel(),
                        y_mean - 1.96 * y_std,
                        y_mean + 1.96 * y_std,
                        alpha=0.2,
                        color='b',
                        label='95% confidence interval')

    # Add labels and legend
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()


def _plotGP2D(gp: GaussianProcessRegressor, xs: np.ndarray, ys: np.ndarray, 
           x_min: float, x_max: float, y_min: float, y_max: float, 
           plot_ci: Optional[bool]=True):
    """
    Plot the Gaussian Process posterior mean and the data points for a 2D problem.

    Parameters
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        The trained GP regressor.
    xs, ys : numpy.ndarray
        The data points.
    x_min, x_max : float
        The lower and upper bounds for the first axis (x-axis) for plotting.
    y_min, y_max : float
        The lower and upper bounds for the second axis (y-axis) for plotting.
    plot_ci : bool, optional
        If True, the confidence interval will be plotted. Default is True.
    """
    assert xs.shape[1] == 2, 'This function only supports 2-dimensional problems'

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
    plt.contourf(x1_grid, x2_grid, y_mean, cmap='viridis', alpha=0.8)
    plt.colorbar(label='GP mean')

    # Plot the original points
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=ys, cmap='viridis', edgecolors='k', marker='o', label='Data points')

    # Plot the confidence interval
    if plot_ci:
        ci_lower = y_mean - 1.96 * y_std
        ci_upper = y_mean + 1.96 * y_std
        plt.contour(x1_grid, x2_grid, ci_lower, levels=1, colors='blue', linestyles='dashed', alpha=0.5)
        plt.contour(x1_grid, x2_grid, ci_upper, levels=1, colors='red', linestyles='dashed', alpha=0.5)

        # Add dummy plots for the legend
        lower_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='blue', alpha=0.5)
        upper_dummy_line = plt.Line2D([0], [0], linestyle='dashed', color='red', alpha=0.5)
        plt.legend([scatter, lower_dummy_line, upper_dummy_line], ['Data points', '95% CI Lower Bound', '95% CI Upper Bound'])

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Gaussian Process Regression for 2D Problems')
    plt.show()