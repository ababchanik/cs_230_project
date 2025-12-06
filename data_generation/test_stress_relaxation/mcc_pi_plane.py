import numpy as np
import matplotlib.pyplot as plt

def plot_iso_mcc_pi(ax, M, pc, p, num=512, label=None, set_equal=True, draw_axes=False, **plot_kwargs):
    """
    Plot the isotropic MCC yield contour in the deviatoric (π) plane for a given mean stress p.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on (e.g., a subplot like axs[1]).
    M : float
        Critical state slope of MCC.
    pc : float
        Preconsolidation pressure (same units as p).
    p : float
        Mean effective (hydrostatic) stress at which to slice the surface.
        Valid slice requires 0 < p < pc.
    num : int, optional
        Number of points used to draw the circle (default 512).
    label : str or None, optional
        Legend label for the curve (default None).
    set_equal : bool, optional
        If True, enforce 1:1 aspect and symmetric limits around the circle (default True).
    draw_axes : bool, optional
        If True, draw π1 and π2 axes through the origin (default False).
    **plot_kwargs :
        Extra keyword args passed to ax.plot (e.g., linewidth=2, linestyle="--", etc.).

    Returns
    -------
    line : matplotlib.lines.Line2D
        The line object for the plotted circle (useful for legends).
    R0 : float
        The circle radius in the π-plane.
    """
    # Radius in π-plane: R0 = sqrt(2/3) * M * sqrt(p * (pc - p))
    val = p * (pc - p)
    if val <= 0:
        # Degenerate slice: no meaningful π-plane contour
        # (either at the tip p<=0 or at/above pc)
        print("pc", pc, "p", p)
        return None, 0.0

    R0 = np.sqrt(2.0/3.0) * M * np.sqrt(val)

    theta = np.linspace(0.0, 2.0*np.pi, num, endpoint=True)
    x = R0 * np.cos(theta)
    y = R0 * np.sin(theta)

    line, = ax.plot(x, y, label=label, **plot_kwargs)

    if set_equal:
        lim = 1.1 * R0
        if draw_axes:
            ax.plot([-lim, lim], [0, 0], linewidth=1)
            ax.plot([0, 0], [-lim, lim], linewidth=1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$\pi_1$")
        ax.set_ylabel(r"$\pi_2$")

    return line, R0


import numpy as np

def show_quadrant_II(ax, pad=0.05):
    """
    Crop an Axes to Quadrant II (x<0, y>0) based on what's already plotted.
    pad is a fractional padding of the data span.
    """
    # collect all line data on this axes
    xs, ys = [], []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if x is None or y is None or len(x) == 0:
            continue
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))
    if not xs:  # nothing plotted
        ax.set_xlim(-1, 0)
        ax.set_ylim(0, 1)
        return

    X = np.concatenate(xs)
    Y = np.concatenate(ys)

    # focus on Q2 if any points exist there; otherwise use all and still crop to Q2
    mask = (X < 0) & (Y > 0)
    if np.any(mask):
        X2, Y2 = X[mask], Y[mask]
    else:
        X2, Y2 = X, Y

    xmin = X2.min()
    xmax = 0.0
    ymin = 0.0
    ymax = Y2.max()

    # padding
    xspan = abs(xmin - xmax)
    yspan = abs(ymax - ymin)
    ax.set_xlim(xmin - pad*xspan, xmax + pad*xspan)
    ax.set_ylim(ymin - pad*yspan, ymax + pad*yspan)
