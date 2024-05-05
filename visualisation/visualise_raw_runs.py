import types

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.image
import matplotlib.colors
import matplotlib.ticker
import matplotlib.patheffects
import matplotlib.pyplot as plt
import re
from typing import Union, Optional

from mpl_toolkits.axes_grid1 import ImageGrid

from utils import loader_utils, plt_utils
from utils.loader_utils import load_run
from classes import Run


pd.options.display.width=1920
pd.options.display.max_columns=99

matplotlib.rc('font', **{
    'family' : 'sans',
    'size'   : 24})

matplotlib.rc("axes",
              titley=1.05)
basepath = Path(__file__).parent



def show_vals(vals: np.ndarray, title: str, ax: plt.Axes, mesh_size: tuple[int,int],
              cbar_title: bool,
              norm: Optional[matplotlib.colors.Normalize] = None,
              add_cb: bool = False) -> matplotlib.image.AxesImage:
    """
    Show values of counters on an AxB mesh on given axis
    Normalise values if norm is given and optionally add a colorbar
    :param vals: counter values
    :param title: title of plot
    :param ax: Axis to plot on
    :param mesh_size: mesh size
    :param cbar_title: whether to add colorbar title
    :param norm: optional normalizer
    :param add_cb: whether to add colorbar
    :returns: AxesImage
    """
    vals = vals.reshape(mesh_size).T

    if isinstance(norm, types.NoneType):
        if not np.any(vals):  # all-zero
            norm = matplotlib.colors.CenteredNorm()
        else:
            norm = matplotlib.colors.LogNorm()
    im = ax.imshow(vals, norm=norm)

    title = title.replace("_dat_txflit_valid", "").replace("mxp_", "")
    ax.set_title(title)
    ax.set_ylim(*ax.get_ylim()[::-1])
    if add_cb:
        plt_utils.add_colorbar(ax=ax, im=im, title="# events" if cbar_title else None)
    return im

def plot_run(run: Run,
             eventids: list[Union[str,re.Pattern]] = None,
             show_remaining: bool = False,
             mesh_size: (int, int) = (8, 6),
             save: bool = False,
             cbar_title: Optional[bool] = None,
             display_meta: bool = True,
             base_size_x: float = 7,
             base_size_y: float = 6,
             ):
    """
    Plot CMN performance counter measurement run as coloured matrix

    Note: If eventids contains a regex pattern, then all measurements matching this pattern will be summed and
           then plotted as one measurement. If no eventids are given, display all events separately.

    :param run: Run instance to plot
    :param eventids: list of eventids as string or regex (default: None -> all events in run)
    :param show_remaining: whether to include event IDs in `run` but not in `eventids` - useful for specifying patterns in `eventids` and including the rest like a wildcard (default: False)
    :param mesh_size: mesh size (default: 8x8)
    :param save: whether to save to disk (default: False)
    :param cbar_title: whether to add colourbar title (default: True if len(eventids)==1)
    :param display_meta: whether to display meta.md in one frame
    :param base_size_x: fig size X per event ID
    :param base_size_y: fig size Y per event ID
    """

    if not isinstance(eventids, list):
        eventids = run.df.event_name.unique()

    if not isinstance(cbar_title, bool):
        cbar_title = len(eventids) == 1

    df = run.df.copy()
    df["counts"] = pd.to_numeric(df["counts"]
                                 .replace("<not counted>", 0)
                                 .replace("<not supported>", -1))
    # check event IDs to plot
    #  if re.Pattern, find and gather all matches
    #  if regular string, add it as-is
    # also keep track of remaining event IDs not matched
    included_eventids = set()
    vals_array = []
    for i, eventid in enumerate(eventids):
        if isinstance(eventid, re.Pattern):
            _events = [e for e in df.event_name.unique() if eventid.match(e)]
            included_eventids |= set(_events)
            vals = np.zeros(shape=(len(df[df.event_name == _events[0]]),))
            for event in _events:
                for _, group in df[df.event_name == event].groupby("node_port"):
                    vals += group["counts"].values
            title = eventid.pattern
        else:
            included_eventids |= {eventid}
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))

            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            title = eventid
        vals_array.append((title, vals))

    # include non-matched event IDs if show_remaining
    if show_remaining:
        for i, eventid in enumerate(list(set(run.df.event_name.unique()) - included_eventids)):
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))
            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            vals_array.append((eventid, vals))

    N = len(vals_array)
    if display_meta: # account for meta pane
        N += 1

    nrows = int(np.ceil(np.sqrt(N)))
    ncols = int(np.ceil(N / nrows))

    data = np.array([tup[1] for tup in vals_array]).flat
    norm = matplotlib.colors.LogNorm(vmin=np.min(list(filter(lambda x: x>0, data))), vmax=np.max(data))

    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size_x*ncols, base_size_y*nrows))
    if nrows == ncols == 1:
        axes = np.array([axes])

    for i, (title, vals) in enumerate(vals_array):
        show_vals(vals=vals, title=title, ax=axes.flat[i], mesh_size=mesh_size, cbar_title=cbar_title,
                  norm=norm, add_cb=i % ncols == ncols - 1)
        axes.flat[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        axes.flat[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    # nrows x ncols might be too much for non-sqrt len(vals_arrays); remove unused axes
    for ax in axes.flat[N:]:
        fig.delaxes(ax)

    # disable yaxis for all but the first axis (save space)
    for vax in axes:
        if not isinstance(vax, plt.Axes):
            for ax in vax[1:]:
                ax.yaxis.set_visible(False)

    if display_meta:
        ax: plt.Axes = axes.flat[-1]
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.text(x=0.1,y=0,s=run.meta, fontdict={"fontsize": 11}, wrap=True)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fig.tight_layout()
    if save: plt.savefig(basepath / "figures" / "raw_runs" / f"{run.name.replace('/','-')}.pdf")
    else:    plt.show()


def plot_run_one_row(run: Run,
                     eventids: list[Union[str,re.Pattern]] = None,
                     show_remaining: bool = False,
                     mesh_size: (int, int) = (8, 6),
                     save: bool = False,
                     suptitle: bool = True,
                     base_size_x: float = 6,
                     base_size_y: float = 6):
    """
    Plot CMN performance counter measurement run as coloured matrix, but keep all event IDs on one row

    :param run: Run instance to plot
    :param eventids: list of eventids as string or regex (default: None -> all events in run)
    :param show_remaining: whether to include event IDs in `run` but not in `eventids` - useful for specifying patterns in `eventids` and including the rest like a wildcard (default: False)
    :param mesh_size: mesh size (default: 8x8)
    :param save: whether to save to disk (default: False)
    :param suptitle: whether to add title
    :param base_size_x: fig size X per event ID
    :param base_size_y: fig size Y per event ID
    """
    if not isinstance(eventids, list):
        eventids = run.df.event_name.unique()

    df = run.df.copy()
    df["counts"] = pd.to_numeric(df["counts"]
                                 .replace("<not counted>", 0)
                                 .replace("<not supported>", -1))

    # check event IDs to plot
    #  if re.Pattern, find and gather all matches
    #  if regular string, add it as-is
    # also keep track of remaining event IDs not matched
    included_eventids = set()
    vals_array = []
    for i, eventid in enumerate(eventids):
        if isinstance(eventid, re.Pattern):
            _events = [e for e in df.event_name.unique() if eventid.match(e)]
            included_eventids |= set(_events)
            vals = np.zeros(shape=(len(df[df.event_name == _events[0]]),))
            for event in _events:
                for _, group in df[df.event_name == event].groupby("node_port"):
                    vals += group["counts"].values
            title = eventid.pattern
        else:
            included_eventids |= {eventid}
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))

            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            title = eventid
        vals_array.append((title, vals))

    # include non-matched event IDs if show_remaining
    if show_remaining:
        for i, eventid in enumerate(list(set(run.df.event_name.unique()) - included_eventids)):
            vals = np.zeros(shape=(mesh_size[0] * mesh_size[1],))
            for _, group in df[df.event_name == eventid].groupby("node_port"):
                vals += group["counts"].values
            vals_array.append((eventid, vals))

    data = np.array([tup[1] for tup in vals_array]).flat
    norm = matplotlib.colors.LogNorm(vmin=np.min(list(filter(lambda x: x>0, data))), vmax=np.max(data))

    # use Image Grid to keep everything tidy
    ncols = len(vals_array)
    fig = plt.figure(figsize=(base_size_x*ncols, base_size_y*1))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, ncols),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15)

    for i, (title, vals) in enumerate(vals_array):
        ax = grid[i]
        im = show_vals(vals, title, ax, mesh_size=mesh_size, norm=norm, add_cb=False, cbar_title=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

        rect = plt.Rectangle(xy=(-.5,-.5), width=8, height=8, hatch="\\", alpha=.5, color="black", zorder=-1, fill=False)
        ax.add_patch(rect)

    cb = grid[-1].cax.colorbar(im)
    cb.set_label("# events")
    grid[0].set_ylabel("Mesh Y")
    grid[len(grid)//2].set_xlabel("Mesh X")


    if suptitle:
        plt.suptitle(f"{run.name}")
    plt.tight_layout(pad=0) # I know, mpl complains, but it works for python 3.11, mpl 3.8.3 & time was short
    if save: plt.savefig(basepath / "figures" / "raw_runs" / f"{run.name.replace('/','-')}.pdf")
    else:    plt.show()


def main():
    basepath_measurements = basepath / "data" / "raw"

    # c2c 60,42
    run_name = "i10se19_2024-02-09T1643"
    plot_run_one_row(load_run(name=run_name, basepath=basepath_measurements,
                              events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
                     eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
                     show_remaining=True, mesh_size=(8, 8), save=True, suptitle=False,
                     base_size_x=4.5, base_size_y=5.5)

    # # same data, but plotted in several rows; also adds content of meta.md
    # plot_run(load_run(name=run_name, basepath=basepath_measurements,
    #                   events=loader_utils.load_events(basepath_measurements / run_name / "events.csv")),
    #          eventids=[ re.compile("mxp_[nesw]_dat_txflit_valid"), ],
    #          show_remaining=True, mesh_size=(8, 8), save=False,
    #          display_meta=True)


if __name__ == "__main__": main()