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

    return fig, ax1

    # if data_file:
    #     _out = pd.DataFrame({"Year": forecast.keys(), _out_col: y0_per_year})
    #
    #     _out.set_index("Year").to_csv(data_file)
    #
    # return fig, ax1, ax2

def plot_forecast_comp(fig, ax, forecast, yavg, ymin, ymax, y0, ylabel,
                  xlabel='Cumulative Capacity, MW', fname=None):
    """
    :param x:
    :param y:
    :param y1:
    :param y2:
    :param y_std:
    :return:
    """



    # Extract annual capacity from forecast
    x = list(forecast.values())
    # Plot
    ax.plot(x, yavg, 'k-')
    ax.plot(x, ymin, 'k--')
    ax.plot(x, ymax, 'k--')
    ax.scatter(x[0]*np.ones(len(y0)), y0, marker='x', c='k')
    # ax1.set_xlabel(xlabel)
    # ax1.set_ylabel(ylabel)
    #
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(forecast.keys(), rotation=45, fontsize=8)

    if fname:
        myformat(ax)
        mysave(fig, fname)
        plt.close()