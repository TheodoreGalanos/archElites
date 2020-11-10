import math
from copy import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams.update({'axes.titlesize': 'small'})

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_heatmap(data,
                 x_axis=None,
                 y_axis=None,
                 v_min=0,
                 v_max=1,
                 title="MapElites fitness map",
                 minimization=True,
                 savefig_path=None,
                 iteration=None,
                 plot_annotations=False,
                 highlight_best=True,
                 interactive=True):

    title = f"{title} - white cells: null values"

    # get data dimensionality
    d = data.shape

    # Show plot annotations just when we have most two dimensions
    # With higher dimensions there would not be enough space
    # if len(d) == 1 or len(d) == 2:
    #     plot_annotations = True

    # reshape data to obtain a 2d heatmap
    if len(d) == 1:
        data = [data]
    if len(d) == 2:
        data = data.transpose()
    if len(d) == 3:
        data = np.transpose(data, axes=(1, 0, 2)).reshape((d[1], d[0] * d[2]))
    if len(d) == 4:
        _data = np.transpose(data, axes=[1, 0, 2, 3])
        data = np.transpose(_data.reshape((d[1], d[0] * d[2], d[3])), axes=[0, 2, 1]).reshape(
            (d[1] * d[3], d[0] * d[2]))

    plt.subplots(figsize=(10, 10))
    copy_data = copy(data)
    df_data = pd.DataFrame(copy_data)
    df_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    mask = df_data.isnull()
    #x_ticks = np.arange(0, 6, 0.16)
    x_ticks = np.arange(0, 63750, 1250)
    #x_ticks = np.arange(0, 104, 4)
    x_ticks = [str(x)[:5] for x in x_ticks]

    #y_ticks = np.arange(0, 0.714, 0.014)
    y_ticks = np.arange(0, 63750, 1250)
    #y_ticks = np.arange(0, 104, 4)
    y_ticks = [str(x)[:5] for x in y_ticks]

    cmap_reversed = matplotlib.cm.get_cmap('YlGnBu_r')
    ax = sns.heatmap(
        df_data,
        vmin=v_min,
        vmax=v_max,
        mask=mask,
        annot=plot_annotations,
        # norm=log_norm,
        fmt=".1f",
        annot_kws={'size': 10},
        # cbar_kws={"ticks": cbar_ticks},
        linewidths=0.5,
        linecolor='grey',
        cmap=cmap_reversed,
        xticklabels=x_ticks,
        yticklabels=y_ticks
    )



    if highlight_best:
        if minimization:
            best = df_data.min().min()
        else:
            best = df_data.max().max()
        title = f"{title} - red cell: best value"
        sns.heatmap(df_data, mask=df_data != best, cmap="Reds_r", annot=plot_annotations, cbar=False, xticklabels=x_ticks, yticklabels=y_ticks)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_title(title)
    ax.invert_yaxis()


    # set ticks
    y_ticks_pos = [0.5]
    x_ticks_pos = range(0, d[0]+1)
    if len(d) > 1:
        y_ticks_pos = range(0, d[1]+1)
    if len(d) > 2:
        x_ticks_pos = range(0, d[0]*d[2]+1, d[2])
    if len(d) > 3:
        y_ticks_pos = range(0, d[1]*d[3]+1, d[3])


    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis))

    #ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks_pos))
    #ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_axis))


    # show grid lines
    thick_grid_color = 'k'
    thick_grid_width = 0.1
    ax.vlines(
        range(0, d[0]),
        *ax.get_xlim(),
        colors=thick_grid_color,
        linewidths=thick_grid_width
        )
    ax.hlines(
        range(0, d[1]),
        *ax.get_ylim(),
        colors=thick_grid_color,
        linewidths=thick_grid_width
        )
    """
    if len(d) == 3:
        ax.vlines(
            list(range(0, d[0] * d[2], d[2])),
            *ax.get_ylim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
        ax.hlines(
            list(range(0, d[1])),
            *ax.get_xlim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
    if len(d) == 4:
        ax.vlines(
            list(range(0, d[0] * d[2] + 1, d[2])),
            *ax.get_ylim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
        ax.hlines(
            list(range(0, d[1] * d[3] + 1, d[3])),
            *ax.get_xlim(),
            colors=thick_grid_color,
            linewidths=thick_grid_width
        )
    """
    #plt.xlabel('Open space considered suitable for long term sitting (m2)', fontsize=10)
    #plt.ylabel('Open space considered dangerous for pedestrians (m2)', fontsize=10)
    plt.xlabel('% of open space that is suitable for long term sitting', fontsize=15)
    plt.ylabel('% of open space that is dangerous for pedestrians', fontsize=15)
    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        ht_figure.savefig(savefig_path / "heatmap_{}.png".format(iteration), dpi=400)
        ht_figure.savefig(savefig_path / "heatmap_{}.pdf".format(iteration), dpi=400)
    if interactive:
        plt.show()
    plt.close()
