import numpy as np

from matplotlib.patches import Rectangle


def plotContainer(ax, container, **kwargs):
    '''
    Plot a container on the provided axes.

    TODO - Args and color schemes.
    '''
    
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
