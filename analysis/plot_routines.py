__author__ = ["Matt Shields", "Jake Nunemaker"]
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Matt Shields"
__email__ = "matt.shields@nrel.gov"
__status__ = "Development"


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.text as txt
import os


def mysave(fig, froot, mode='png'):
    assert mode in ['png', 'eps', 'pdf', 'all']
    fileName, fileExtension = os.path.splitext(froot)
    padding = 0.1
    dpiVal = 200
    legs = []
    for a in fig.get_axes():
        addLeg = a.get_legend()
        if not addLeg is None: legs.append(a.get_legend())
    ext = []
    if mode == 'png' or mode == 'all':
        ext.append('png')
    if mode == 'eps':  # or mode == 'all':
        ext.append('eps')
    if mode == 'pdf' or mode == 'all':
        ext.append('pdf')

    for sfx in ext:
        fig.savefig(fileName + '.' + sfx, format=sfx, pad_inches=padding, bbox_inches='tight',
                    dpi=dpiVal, bbox_extra_artists=legs)


titleSize = 24  # 40 #38
axLabelSize = 20  # 38 #36
tickLabelSize = 18  # 30 #28
legendSize = tickLabelSize + 2
textSize = legendSize - 2
deltaShow = 4


def myformat(ax, mode='save'):
    assert type(mode) == type('')
    assert mode.lower() in ['save', 'show'], 'Unknown mode'

    def myformat(myax):
        if mode.lower() == 'show':
            for i in myax.get_children():  # Gets EVERYTHING!
                if isinstance(i, txt.Text):
                    i.set_size(textSize + 3 * deltaShow)

            for i in myax.get_lines():
                if i.get_marker() == 'D': continue  # Don't modify baseline diamond
                i.set_linewidth(4)
                # i.set_markeredgewidth(4)
                i.set_markersize(10)

            leg = myax.get_legend()
            if not leg is None:
                for t in leg.get_texts(): t.set_fontsize(legendSize + deltaShow + 6)
                th = leg.get_title()
                if not th is None:
                    th.set_fontsize(legendSize + deltaShow + 6)

            myax.set_title(myax.get_title(), size=titleSize + deltaShow, weight='bold')
            myax.set_xlabel(myax.get_xlabel(), size=axLabelSize + deltaShow, weight='bold')
            myax.set_ylabel(myax.get_ylabel(), size=axLabelSize + deltaShow, weight='bold')
            myax.tick_params(labelsize=tickLabelSize + deltaShow)
            myax.patch.set_linewidth(3)
            for i in myax.get_xticklabels():
                i.set_size(tickLabelSize + deltaShow)
            for i in myax.get_xticklines():
                i.set_linewidth(3)
            for i in myax.get_yticklabels():
                i.set_size(tickLabelSize + deltaShow)
            for i in myax.get_yticklines():
                i.set_linewidth(3)

        elif mode.lower() == 'save':
            for i in myax.get_children():  # Gets EVERYTHING!
                if isinstance(i, txt.Text):
                    i.set_size(textSize)

            for i in myax.get_lines():
                if i.get_marker() == 'D': continue  # Don't modify baseline diamond
                i.set_linewidth(4)
                # i.set_markeredgewidth(4)
                i.set_markersize(10)

            leg = myax.get_legend()
            if not leg is None:
                for t in leg.get_texts(): t.set_fontsize(legendSize)
                th = leg.get_title()
                if not th is None:
                    th.set_fontsize(legendSize)

            myax.set_title(myax.get_title(), size=titleSize, weight='bold')
            myax.set_xlabel(myax.get_xlabel(), size=axLabelSize, weight='bold')
            myax.set_ylabel(myax.get_ylabel(), size=axLabelSize, weight='bold')
            myax.tick_params(labelsize=tickLabelSize)
            myax.patch.set_linewidth(3)
            for i in myax.get_xticklabels():
                i.set_size(tickLabelSize)
            for i in myax.get_xticklines():
                i.set_linewidth(3)
            for i in myax.get_yticklabels():
                i.set_size(tickLabelSize)
            for i in myax.get_yticklines():
                i.set_linewidth(3)

    if type(ax) == type([]):
        for i in ax: myformat(i)
    else:
        myformat(ax)


def initFigAxis():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    return fig, ax

def scatter_plot(x, y, myxlabel, myylabel, fname=None):
    fig, ax = initFigAxis()

    ax.scatter(x, y, c='k')
    ax.set_xlabel(myxlabel)
    ax.set_ylabel(myylabel)

    if fname:
        myformat(ax)
        mysave(fig, fname)
        plt.close()

    return ax

def plot_forecast_v1(
    installed,
    capex,
    capex_std,
    fit,
    forecast,
    bse=None,
    perc_change=False,
    data_file=None,
    fname=None,
    **kwargs,
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
    axes : list of matplotlib.Axis
    perc_change : bool
    data_file :
    """


    fig, ax1 = initFigAxis()
    ax2 = ax1.twiny()

    upcoming = [v - installed for _, v in forecast.items()]

    x = np.linspace(installed, upcoming[-1])
    b0 = fit
    C0_0 = capex / (installed ** b0)

    if perc_change is False:
        y0 = calc_curve(x, C0_0, b0)
        y0_per_year = calc_curve(upcoming, C0_0, b0)
        _out_col = "Average global CapEx, $/KW"

    else:
        y0 = calc_curve(x, C0_0, b0, capex_0=capex)
        y0_per_year = calc_curve(upcoming, C0_0, b0, capex_0=capex)
        _out_col = "Percent change from initial CapEx"

    ax1.plot(x, y0, "k-")
    ax1.errorbar(x[0], y0[0], capex_std, marker='o', markerfacecolor='none', mec='k', mew=3,
                 ecolor='k', elinewidth=3, capsize=10, capthick=3)
    ax1.set_xlabel("Cumulative Capacity")
    ax1.set_ylabel("CAPEX, $/KW")


    b1 = fit + bse
    b2 = fit - bse

    C0_hi = (capex + capex_std) / (installed ** b1)  # Higher initial costs, lower learning rate
    C0_lo = (capex - capex_std) / (installed ** b2)  # Lower initial costs, higher learning rate

    if perc_change is False:
        y1 = calc_curve(x, C0_hi, b1)
        y2 = calc_curve(x, C0_lo, b2)
    else:
        y1 = calc_curve(x, C0_hi, b1, capex_0=capex+capex_std)
        y2 = calc_curve(x, C0_lo, b2, capex_0=capex-capex_std)
    ax1.fill_between(x, y1, y2, alpha=0.5)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(upcoming)
    ax2.set_xticklabels(forecast.keys(), rotation=45, fontsize=8)
    ax2.set_ylabel("Projected COD")

    if fname:
        myformat([ax1, ax2])
        mysave(fig, fname)
        plt.close()

    if data_file:
        _out = pd.DataFrame({"Year": forecast.keys(), _out_col: y0_per_year})

        _out.set_index("Year").to_csv(data_file)

    return fig, ax1, ax2

def calc_curve(x, C0, b, capex_0=None):
    """Fit the learning curve to a prescribed range of years"""
    if capex_0:
        """Determine percent change from initial capex value"""
        y = 1 - C0 * x ** b / capex_0
    else:
        y = C0 * x ** b

    return y

def plot_forecast(forecast, y, y1, y2, y_std, ylabel,
                  xlabel='Cumulative Capacity, MW', fname=None):
    """
    :param x:
    :param y:
    :param y1:
    :param y2:
    :param y_std:
    :return:
    """

    fig, ax1 = initFigAxis()
    ax2 = ax1.twiny()

    # Extract annual capacity from forecast
    x = list(forecast.values())
    # Plot
    ax1.plot(x, y, 'k-')
    ax1.errorbar(x[0], y[0], y_std, marker='o', markerfacecolor='none', mec='k', mew=3,
                 ecolor='k', elinewidth=3, capsize=10, capthick=3)
    ax1.fill_between(x, y1, y2, alpha=0.5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(forecast.keys(), rotation=45, fontsize=8)

    if fname:
        myformat([ax1, ax2])
        mysave(fig, fname)
        plt.close()

    # if data_file:
    #     _out = pd.DataFrame({"Year": forecast.keys(), _out_col: y0_per_year})
    #
    #     _out.set_index("Year").to_csv(data_file)
    #
    # return fig, ax1, ax2

