"""
Plotting utilities for triangular mesh fields.

Provides the `plot` function for visualizing spatial fields on triangular
meshes using tricontourf, with optional saving to file.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import shutil

# Use LaTeX for text rendering when available, otherwise fall back to
# matplotlib's default mathtext renderer (avoids failures when latex is
# not installed on the system).
if shutil.which("latex") is not None:
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['image.cmap'] = 'RdBu_r'


def plot(triang, list_fields, list_labels,
         vmin=None, vmax=None, filename=None,
         figsize=None, **kwargs):
    """
    Plot one or more fields on a triangular mesh using tricontourf.

    Each subplot receives its own colorbar.

    Args:
        triang: Triangulation object for the mesh.
        list_fields: List of field arrays to plot.
        list_labels: List of labels for each field.
        vmin: Minimum value for color normalization (optional).
        vmax: Maximum value for color normalization (optional).
        filename: If provided, save the figure to this path.
        figsize: Figure size tuple (optional).
        **kwargs: Additional keyword arguments passed to tricontourf.
    """
    n = len(list_fields)
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin - 0.01 * np.abs(vmin),
                             vmax + 0.01 * np.abs(vmax), 20)
        if vmin < 0 < vmax:
            norm = matplotlib.colors.TwoSlopeNorm(0, vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        levels = 20
        norm = None

    if figsize is None:
        figsize = (4 * n + 0.8 * n, 3) if n > 1 else (5, 3)

    fig, axs = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for field, label, ax in zip(list_fields, list_labels, axs[0]):
        contour_kwargs = dict(levels=levels, **kwargs)
        if norm is not None:
            contour_kwargs["norm"] = norm
        mappable = ax.tricontourf(triang, field, **contour_kwargs)
        ax.set_title(label)
        fig.colorbar(mappable, ax=ax)

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
