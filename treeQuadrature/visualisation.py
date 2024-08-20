import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.axes import Axes

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
                   dimensions: Optional[List[float]]=None, 
                   colors: str='YlOrRd',
                   c_bar_labels: str='Contributions'):
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
    colors : str, Optional
        the colour map used to plot 2D 
        container contributions.
        Default : 'YlOrRd'
    c_bar_labels : str, Optional
        labels for colour bar in 
        2D plot. 
        Default : 'Contributions'
    """

    assert len(containers) == len(contributions), (
        'The length of containers and contributions must be the same, '
        f'got {len(containers)} containers and {len(contributions)} contributions'
    )

    # check dimensions
    all_dimensions = list(range(containers[0].D))
    if dimensions is None:
        dimensions = all_dimensions
    elif len(dimensions) != 2:
        raise ValueError('dimensions must have length 2')
    
    if len(all_dimensions) > 1:
        assert (dimensions[0] in all_dimensions) and (
            dimensions[1] in all_dimensions), (
                f'dimensions must be in 0, 1, ..., {len(all_dimensions)-1}'
            )

    if containers[0].D ==1:
        _plotContainers1D(containers, contributions, xlim, 
                          integrand, title, plot_samples)
    elif len(dimensions) == 2:
        _plotContainers2D(containers, contributions, xlim, ylim, title, 
                          plot_samples, dimensions[0], dimensions[1], 
                          colors, c_bar_labels)
    else:
        raise ValueError(
            "Only 1D and 2D plots are supported. "
             "Please provide 2 dimensions to plot for higher dimensions"
             )


def _plotContainers2D(containers, contributions, xlim, ylim, title, 
                      plot_samples, dim1, dim2, colors, c_bar_labels):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = colormaps[colors].resampled(256)

    # if len(contributions) > 1:
    #     contributions = scale(contributions)

    norm = plt.Normalize(min(contributions), max(contributions))

    for container, contribution in zip(containers, contributions):
        plotContainer(ax, container, dim1, dim2, 
                      plot_samples=plot_samples, facecolor=cmap(norm(contribution)), alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(c_bar_labels)

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
        and any other kwargs for matplotlib.patches.Rectangle
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

def plotIntegrand(integrand, D, xlim, ylim=None, n_points=500, levels=10, 
                  file_path: Optional[str]=None):
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
    file_path : str, optional
        If given, the figure will be saved to 
        the specified path. 
    """

    if D == 1:
        _plot1D(integrand, xlim, n_points, file_path)
    elif D == 2:
        _plot2D(integrand, xlim, ylim, levels, n_points, file_path)
    else:
        raise Exception('only supports 1D and 2D problems')

def _plot1D(f, xlim, n_points, file_path):
    x = np.linspace(xlim[0], xlim[1], n_points).reshape(-1, 1)
    ys = f(x)

    plt.plot(x, ys)
    plt.xlabel('x')
    plt.ylabel('Value')
    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f'figure saved to {file_path}')
    else:
        plt.show()

def _plot2D(f, xlim, ylim, levels, n_points, file_path):
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

    if file_path:
        plt.savefig(file_path)
        plt.close()
        print(f'figure saved to {file_path}')
    else:
        plt.show()


def plot_errors(data: pd.DataFrame, genres: List[str], 
                error_bar: bool = False, plot_absolute=False, 
                plot_all_errors: bool = False, 
                fill: bool = False,
                y_lim: Optional[List]=None, 
                font_size: int = 10,
                offset: float = 0.1,
                plot_title: bool = True, 
                grid: bool = True, 
                filename_prefix: Optional[str] = None, 
                integrators: Optional[List[str]] = None) -> None:
    """
    Plot errors and error_std for each genre and integrator
    and save it to a file figures/filename_prefix_genre_error_plot.csv. 

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the results of the integrators.
    genres : list of str
        List of genres to include in the plots.
    error_bar : bool, optional
        If True, plot error bars; otherwise plot without error bars.
        Default is False
    fill : bool, optional
        If true, fill between the error bars, 
        will be ignored if error_bar = False.  
        Default is False
    plot_all_errors : bool, optional
        If True, plot all individual errors from 'errors' column.
        Default is False
    plot_absolute : bool, optional
        If True and error_type is 'Signed Relative error', 
        plot the signed absolute errors.
        Default is False
    y_lim: list of float, optional
        the upper and lower limit for plotting error. 
        If not given, adaptive limits applied. 
    font_size : int, optional
        Font size for all text elements in the plot (default is 10).
    offset: float, optional
        offset of points to avoid visual clutter.
        Default is 0.1. 
        Set to 0 for no offset
    plot_title : bool, optional
        whether add title or not (default is True).
    grid : bool, optional
        whether to plot the grid or not. 
        Default is True. 
    filename_prefix : str, optional
        The prefix for the filenames where plots will be saved.
        If not given, no prefix, 
    integrators : List[str], optional
        List of integrators to include in the plots. 
        If not specified, all integrators will be plotted

    Notes
    -----
    The data should contain columns 'problem', 'integrator', 'error', 
    'error_std', and 'errors'. The 'error' and 'error_std' 
    columns are expected to have percentage values.

    Usage
    -----
    >>> all_data = pd.read_csv('your_data.csv')  # Load your data into a DataFrame
    >>> genres = ["SimpleGaussian", "Camel", "QuadCamel"]
    >>> plot_errors(all_data, genres)
    """

    all_integrators = data['integrator'].unique()

    # use all integrators if not specified
    if integrators is None:
        integrators = all_integrators
    else:
        for integrator in integrators:
            if integrator not in all_integrators:
                raise ValueError(f"Integrator {integrator} not found in data")

    plt.rcParams.update({'font.size': font_size})

    legend_font_size = font_size - 3
    axis_label_font_size = font_size + 3 

    data['Dimension'] = data['problem'].str.extract(r'D=(\d+)').astype(int)

    # Ensure the 'error' and 'error_std' columns are numeric
    data['error'] = data['error'].str.replace('%', '').astype(float)
    data['error_std'] = data['error_std'].str.replace('%', '').astype(float)

    # Define a color map to ensure consistent colors for each integrator
    color_map = plt.get_cmap('tab10')
    color_dict = {integrator: color_map(i) for i, integrator in enumerate(integrators)}

    offsets = np.linspace(-offset, offset, len(integrators))

    for genre in genres:
        genre_data = data[data['problem'].str.contains(genre, na=False, case=False)]

        if genre_data.empty:
            print(f"Genre {genre} not found in the 'problem' column")
            continue

        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 10))

        max_error = -np.inf
        min_error = np.inf
        
        for idx, integrator in enumerate(integrators):
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
                if isinstance(error_list_str, float) and np.isnan(error_list_str):
                    continue
                else:
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
            x_offset = np.array(used_dimensions) + offsets[idx]
            if plot_all_errors:
                first_label = True
                for dim, all_errors in zip(used_dimensions, all_errors_list):
                    label = f'{integrator} All Errors' if first_label else None
                    plt.scatter([dim + offsets[idx]] * len(all_errors), all_errors, alpha=0.3, 
                                label=label, color=color)
                    first_label = False
            elif error_bar:
                lower_bound = [e - es for e, es in zip(errors, error_stds)]
                upper_bound = [e + es for e, es in zip(errors, error_stds)]
                if fill:
                    plt.fill_between(x_offset, lower_bound, upper_bound, 
                                    alpha=0.2, label=f'{integrator} Std')
                else:
                    plt.errorbar(x_offset, errors, yerr=error_stds,
                                 fmt='o', capsize=5)    

            plt.plot(x_offset, errors, label=integrator, marker='o', color=color)
        
        if y_lim is not None:
            plt.ylim(y_lim)
        elif plot_absolute:
            plt.ylim([min_error - 0.05 * np.abs(min_error), 
                      max_error + 0.05 * np.abs(max_error)])
        else:
            plt.ylim([-105, 105])
        
        if plot_title:
            plt.title(f'Error and Error Std for {genre}')
        plt.xlabel('Dimension', fontsize=axis_label_font_size)
        if plot_absolute:
            plt.ylabel('Absolute Error', fontsize=axis_label_font_size)
        else:
            plt.ylabel(f'{data["error_type"].values[0]} (%)', 
                       fontsize=axis_label_font_size)
        plt.ylim(y_lim)
        plt.legend(fontsize=legend_font_size)
        plt.grid(grid)
        if filename_prefix:
            plt.savefig(f'figures/{filename_prefix}{genre}_error_plot.png')
            plt.close()
            print(f'Figure saved to figures/{filename_prefix}{genre}_error_plot.png')
        else:
            plt.savefig(f'figures/{genre}_error_plot.png')
            plt.close()
            print(f'Figure saved to figures/{genre}_error_plot.png')


def plot_times(data: pd.DataFrame, genres: List[str], 
               font_size: int = 10, filename_prefix: Optional[str] = None, 
               integrators: Optional[list]=None, title: bool=True) -> None:
    """
    Plot the time taken for each genre and integrator, 
    and save it to a file 
    figures/filename_prefix_genre_time_plot_integrator.csv. 
    Used for csv files produced by test_integrators
    
    Parameters
    ----------
    data : pd.DataFrame
        The data containing the results of the integrators.
    genres : list of str
        List of problem genres to include in the plots.
    font_size : int, optional
        Font size for all text elements in the plot (default is 10).
    filename_prefix : str, Optional
        The prefix for the filenames where plots will be saved.
        If not given, no prefix
    integrators : List[str], optional
        List of integrators to include in the plots. 
        If not specified, all integrators will be plotted
    title : bool, optional
        If true, plot the title

    Notes
    -----
    The data should contain columns 'problem', 'integrator', and 'time_taken'

    Usage
    -----
    >>> all_data = pd.read_csv('your_data.csv')  # Load your data into a DataFrame
    >>> genres = ["SimpleGaussian", "Camel", "QuadCamel"]
    >>> plot_times(all_data, 'figures/time', genres)
    """
    plt.rcParams.update({'font.size': font_size})

    all_integrators = data['integrator'].unique()

    # use all integrators if not specified
    if integrators is None:
        integrators = all_integrators
    else:
        for integrator in integrators:
            if integrator not in all_integrators:
                raise ValueError(f"Integrator {integrator} not found in data")

    data['Dimension'] = data['problem'].str.extract(r'D=(\d+)').astype(int)

    for genre in genres:
        genre_data = data[data['problem'].str.contains(genre)]
        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 10))
    
        for integrator in integrators:
            genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
            times = []
            
            for dim in dimensions:
                data_dim = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
                if not data_dim.empty and data_dim['time_taken'].values[0] != '':
                    time = data_dim['time_taken'].values[0]
                    try:
                        time = float(time)
                        times.append(time)
                    except ValueError:
                        times.append(float('nan'))
                else:
                    times.append(float('nan'))
            
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.plot(dimensions, times, marker='o')
            if title:
                plt.title(f'Time Taken for {genre} \n {integrator}')
            plt.xlim(min(dimensions)-1, max(dimensions)+1)
            plt.xlabel('Dimension')
            plt.ylabel('Time Taken (seconds)')
            plt.grid(True)
            plt.tight_layout()
            if filename_prefix:
                plt.savefig(f'figures/{filename_prefix}_{genre}_time_plot_{integrator}.png')
                plt.close()
                print(f'Figure saved to figures/{filename_prefix}_{genre}_time_plot_{integrator}.png')
            else:
                plt.savefig(f'figures/{genre}_time_plot_{integrator}.png')
                plt.close()
                print(f'Figure saved to figures/{genre}_time_plot_{integrator}.png')