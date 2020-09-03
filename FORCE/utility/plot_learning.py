__author__ = ["Jake Nunemaker", "Matt Shields", "Philipp Beiter"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


import numpy as np
import matplotlib.pyplot as plt


def plot_learning_forecast(
    installed, capex, fit, forecast, bse=None, axes=None, **kwargs
):
    """
    Plots forecasted CAPEX/kW based on the installed capacity, current capex,
    fit parameters and the forecasted cumulative capacity.

    Parameters
    ----------
    installed : float
        Installed capacity at start of forecast (MW).
    capex: float
        CAPEX at start of forecast ($/kW)
    fit : float
    forecast : dict
        Dictionary of forecasted capacity with format:
        'year': 'MW of capacity'.
    bse : float | None
        Standard error of the fit.
        If None, error will not be plotted.
    axes : matplotlib.Axis
    """

    if axes is None:
        fig = plt.figure(**kwargs)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

    else:
        raise NotImplementedError(
            "Passing in pre-constructed axes is not supported yet."
        )

    upcoming = [v - installed for _, v in forecast.items()]

    x = np.linspace(installed, upcoming[-1])
    b0 = fit
    C0_0 = capex / (installed ** b0)
    y0 = C0_0 * x ** b0

    ax1.plot(x, y0, "k-")
    ax1.set_xlabel("Cumulative Capacity")
    ax1.set_ylabel("CAPEX, $/KW")

    if bse:
        b1 = fit + bse
        b2 = fit - bse

        C0_1 = capex / (installed ** b1)
        C0_2 = capex / (installed ** b2)

        y1 = C0_1 * x ** b1
        y2 = C0_2 * x ** b2

        ax1.fill_between(x, y1, y2)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(upcoming)
    ax2.set_xticklabels(forecast.keys(), rotation=45, fontsize=8)
    ax2.set_ylabel("Projected COD")

    return ax1, ax2
