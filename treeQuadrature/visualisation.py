from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

from treeQuadrature.utils import scale

def plotContainers(containers, contributions, xlim, ylim=None, integrand=None, 
                   title=None, plot_samples=False):
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
    """

    assert len(containers) == len(contributions), (
        'The length of containers and contributions must be the same'
        f'got {len(containers)} and {len(contributions)}'
    )

    # find the dimension
    D = containers[0].X.shape[1]

    if D == 2:
        _plotContainers2D(containers, contributions, xlim, ylim, title, 
                          plot_samples=plot_samples)
    elif D == 1:
        _plotContainers1D(containers, contributions, xlim, integrand, title, 
                          plot_samples=plot_samples)
    else:
        raise Exception('only supports 1D and 2D problems')


def _plotContainers2D(containers, contributions, xlim, ylim, title, plot_samples):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = colormaps['YlOrRd'].resampled(256)

    contributions = scale(contributions)

    for container, contribution in zip(containers, contributions):
        plotContainer(ax, container, plot_samples=plot_samples, facecolor=cmap(contribution), alpha=0.4)

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

def plotContainer(ax, container, **kwargs):
    '''
    Plot a container on the provided axes.

    Parameters
    --------
    ax: Axes
        the canvas
    container : Container
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

    # check dimension
    assert container.X.shape[1] == 2, 'plotContainer only supports two dimensional problem'

    # kwargs
    plot_samples = kwargs.pop('plot_samples', True)
    s = kwargs.pop('s', 5.0)
    fill = kwargs.pop('fill', True)
    ec = kwargs.pop('ec', 'black')

    # Plotting container samples
    if plot_samples:
        ax.scatter(container.X[:, 0], container.X[:, 1],
                   color='navy', s=s, alpha=0.3)

    # Plot container boundary
    x1, y1 = container.mins[:2]
    x2, y2 = container.maxs[:2]
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

def plotGP(gp, xs, ys, x_min, x_max, plot_ci=True):
    """
    Plot the Gaussian Process posterior mean
    and the data points

    Parameters
    ----------
    gp : GaussianProcessRegressor
        the trained GP regressor
    xs, ys : numpy arrays
        the data points
    x_min, x_max : floats
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