import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.axes import Axes
from sklearn.gaussian_process import GaussianProcessRegressor

import pandas as pd

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


def plot_errors(data: pd.DataFrame, filename_prefix: str, genres: List[str], 
                error_bar: bool = False, plot_absolute=False, 
                plot_all_errors: bool = False, 
                y_lim: Optional[List]=None):
    """
    Plot errors and error_std for each genre and integrator.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the results of the integrators.
    filename_prefix : str
        The prefix for the filenames where plots will be saved.
    genres : list of str
        List of genres to include in the plots.
    error_bar : bool, optional
        If True, plot error bars; otherwise plot without error bars.
    plot_all_errors : bool, optional
        If True, plot all individual errors from 'errors' column.
    plot_absolute : bool, optional
        If True and error_type is 'Signed Relative error', plot the signed absolute errors.
    y_lim: list of float, optional
        the upper and lower limit for plotting error

    Notes
    -----
    The data should contain columns 'problem', 'integrator', 'error', 
    'error_std', and 'errors'. The 'error' and 'error_std' 
    columns are expected to have percentage values.

    Usage
    -----
    >>> all_data = pd.read_csv('your_data.csv')  # Load your data into a DataFrame
    >>> genres = ["SimpleGaussian", "Camel", "QuadCamel"]
    >>> plot_errors(all_data, 'figures/error', genres)
    """

    data['Dimension'] = data['problem'].str.extract(r'D=(\d+)').astype(int)

    # Ensure the 'error' and 'error_std' columns are numeric
    data['error'] = data['error'].str.replace('%', '').astype(float)
    data['error_std'] = data['error_std'].str.replace('%', '').astype(float)

    # Define a color map to ensure consistent colors for each integrator
    color_map = plt.get_cmap('tab10')
    integrators = data['integrator'].unique()
    color_dict = {integrator: color_map(i) for i, integrator in enumerate(integrators)}
    
    for genre in genres:
        genre_data = data[data['problem'].str.contains(genre)]
        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 7))

        max_error = -np.inf
        min_error = np.inf
        
        for integrator in genre_data['integrator'].unique():
            if integrator == 'Vegas':
                continue
            genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
            errors = []
            error_stds = []
            all_errors_list = []
            used_dimensions = []
            
            for dim in dimensions:
                data_dim = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
                if data_dim.empty or pd.isnull(data_dim['errors']).all():
                    continue

                error_list_str = data_dim['errors'].values[0]
                error_list = list(map(float, error_list_str.strip('[]').split()))
                error_std = float(data_dim['error_std'].values[0])
                if plot_absolute and (
                    data_dim['error_type'].values[0] == 'Signed Relative error'):
                    true_value = float(data_dim['true_value'].values[0])
                    error_list = true_value * np.array(error_list) / 100.0
                    error_std = true_value * error_std / 100.0

                errors.append(np.median(error_list))
                error_stds.append(error_std)
                all_errors_list.append(error_list)
                used_dimensions.append(dim)

                # for defining y-axis limits
                max_error = max(max_error, np.max(error_list) + error_std)
                min_error = min(min_error, np.min(error_list) - error_std)

            
            color = color_dict[integrator]
            if plot_all_errors:
                first_label = True
                for dim, all_errors in zip(used_dimensions, all_errors_list):
                    label = f'{integrator} All Errors' if first_label else None
                    plt.scatter([dim] * len(all_errors), all_errors, alpha=0.3, 
                                label=label, color=color)
                    first_label = False
            elif error_bar:
                lower_bound = [e - es for e, es in zip(errors, error_stds)]
                upper_bound = [e + es for e, es in zip(errors, error_stds)]
                plt.fill_between(used_dimensions, lower_bound, upper_bound, 
                                 alpha=0.2, label=f'{integrator} Range')
                plt.plot(used_dimensions, errors, label=integrator, marker='o')
            else:
                plt.plot(used_dimensions, errors, label=integrator, marker='o')
        
        if y_lim is not None:
            plt.ylim(y_lim)
        elif plot_absolute:
            plt.ylim([min_error - 0.05 * np.abs(min_error), 
                      max_error + 0.05 * np.abs(max_error)])
        else:
            plt.ylim([-105, 105])

        plt.title(f'Error and Error Std for {genre}')
        plt.xlabel('Dimension')
        if plot_absolute:
            plt.ylabel('Absolute Error')
        else:
            plt.ylabel(f'{data["error_type"].values[0]} (%)')
        plt.ylim(y_lim)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/{filename_prefix}_{genre}_error_plot.png')
        plt.close()
        print(f'Figure saved to figures/{filename_prefix}_{genre}_error_plot.png')


def plot_times(data: pd.DataFrame, filename_prefix: str, genres: List[str]):
    """
    Plot the time taken for each genre and integrator.
    used for csv files produced by test_integrators
    
    Parameters
    ----------
    data : pd.DataFrame
        The data containing the results of the integrators.
    filename_prefix : str
        The prefix for the filenames where plots will be saved.
    genres : list of str
        List of genres to include in the plots.

    Notes
    -----
    The data should contain columns 'problem', 'integrator', and 'time_taken'

    Usage
    -----
    >>> all_data = pd.read_csv('your_data.csv')  # Load your data into a DataFrame
    >>> genres = ["SimpleGaussian", "Camel", "QuadCamel"]
    >>> plot_times(all_data, 'figures/time', genres)
    """

    data['Dimension'] = data['problem'].str.extract(r'D=(\d+)').astype(int)

    for genre in genres:
        genre_data = data[data['problem'].str.contains(genre)]
        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 7))
    
        for integrator in genre_data['integrator'].unique():
            genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
            times = []
            
            for dim in dimensions:
                data_dim = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
                if not data_dim.empty:
                    times.append(float(data_dim['time_taken'].values[0]))
                    
            plt.plot(dimensions, times, label=integrator, marker='o')

            plt.title(f'Time Taken for {genre} - {integrator}')
            plt.xlabel('Dimension')
            plt.ylabel('Time Taken (seconds)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'figures/{filename_prefix}_{genre}_time_plot_{integrator}.png')
            plt.close()
            print(f'Figure saved to figures/{filename_prefix}_{genre}_time_plot_{integrator}.png')